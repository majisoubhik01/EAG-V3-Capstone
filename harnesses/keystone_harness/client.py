"""Guarded MCP transport with explicit hosted read and disposable-write opt-ins."""

from __future__ import annotations

import ipaddress
import json
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from weakref import WeakSet

import requests

from .config import KeystoneConfig, KeystoneConfigurationError
from .discovery import (
    CollectionEvidence,
    CompletenessContract,
    CompletenessResult,
    CompletenessStatus,
    PageEvidence,
    validate_completeness,
)

if TYPE_CHECKING:
    from .calendar_list_transport import CalendarListPilotTransport
    from .mcp_response import ParsedMcpToolCallResponse


class KeystoneClientError(RuntimeError):
    """Base class for sanitized Keystone client failures."""


class KeystoneAuthenticationError(KeystoneClientError):
    """Raised when an authenticated operation has no Keystone token."""


class KeystoneOperationNotAllowedError(KeystoneClientError):
    """Raised when an operation is not explicitly registered as read-only."""


class KeystoneTimeoutError(KeystoneClientError):
    """Raised when the Keystone request exceeds its configured timeout."""


class KeystoneTransportError(KeystoneClientError):
    """Raised for transport failures without exposing request diagnostics."""


class KeystoneHttpError(KeystoneClientError):
    """Raised for non-success HTTP responses without exposing response bodies."""


class KeystoneJsonError(KeystoneClientError):
    """Raised when a response body is not valid JSON."""


class KeystoneProtocolError(KeystoneClientError):
    """Raised when the response does not satisfy the JSON-RPC response shape."""


class KeystoneIncompleteDiscoveryError(KeystoneClientError):
    """Raised when tool discovery indicates more data than this client can fetch."""

    def __init__(
        self,
        completeness: CompletenessResult,
        collection_evidence: CollectionEvidence,
    ) -> None:
        self.completeness = completeness
        self.collection_evidence = collection_evidence
        super().__init__(
            "Keystone tools/list discovery is incomplete; continuation is not implemented."
        )


class KeystoneRpcError(KeystoneClientError):
    """Raised for JSON-RPC errors without echoing server-provided text."""


class KeystoneToolError(KeystoneClientError):
    """Raised when an MCP result reports an operation error."""


@dataclass(frozen=True)
class DiscoveredTool:
    """Non-executable metadata returned by MCP tool discovery."""

    name: str
    read_only_hint: bool | None
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolDiscoveryResult:
    """One tools/list response and its response-scoped completeness evidence."""

    tools: tuple[DiscoveredTool, ...]
    collection_evidence: CollectionEvidence
    completeness: CompletenessResult
    global_catalog_completeness_proven: bool = field(default=False, init=False)


@dataclass(frozen=True, eq=False)
class KeystoneReadAuthorization:
    """Client-bound local policy decision; it is not an executable capability."""

    operation: str
    _arguments_json: str = field(repr=False)
    _client_identity: object = field(repr=False)
    _config: KeystoneConfig = field(repr=False)
    _session: object = field(repr=False)
    _discovery: ToolDiscoveryResult = field(repr=False)
    _descriptor_snapshot: DiscoveredTool = field(repr=False)

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self._arguments_json)


@dataclass(frozen=True, eq=False)
class KeystoneMutationAuthorization:
    """Client-bound opt-in authorization for one disposable Calendar mutation."""

    operation: str
    _arguments_json: str = field(repr=False)
    _client_identity: object = field(repr=False)
    _config: KeystoneConfig = field(repr=False)
    _session: object = field(repr=False)
    _discovery: ToolDiscoveryResult = field(repr=False)
    _descriptor_snapshot: DiscoveredTool = field(repr=False)

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self._arguments_json)


@dataclass(frozen=True)
class _CalendarListPilotRequest:
    """Client-claimed immutable request state passed only to the bound transport."""

    _client_identity: object = field(repr=False)
    _transport_identity: object = field(repr=False)
    endpoint: str
    timeout: tuple[float, float]
    request_id: int
    output_schema: dict[str, Any] | None = field(repr=False)


# Generic call_read remains disabled; hosted reads require a separate opt-in.
PRODUCTION_READ_ONLY_OPERATIONS: tuple[str, ...] = ()
_HOSTED_MCP_PROTOCOL_VERSION = "2025-11-25"


class KeystoneClient:
    """Keystone client with discovery-only defaults and explicit hosted opt-ins.

    ``tools/list`` discovers metadata. Raw ``call_read`` stays disabled;
    hosted reads and disposable Calendar mutations use separate authorization
    paths, both disabled by default.
    """

    def __init__(
        self,
        config: KeystoneConfig,
        session: requests.Session | None = None,
        *,
        enable_hosted_read_transport: bool = False,
        enable_hosted_mutation_transport: bool = False,
        approved_disposable_base_url: str | None = None,
    ) -> None:
        if not isinstance(config, KeystoneConfig):
            raise TypeError("config must be a KeystoneConfig")
        if type(enable_hosted_read_transport) is not bool:
            raise TypeError("enable_hosted_read_transport must be a boolean")
        if type(enable_hosted_mutation_transport) is not bool:
            raise TypeError("enable_hosted_mutation_transport must be a boolean")
        if approved_disposable_base_url is not None and not isinstance(
            approved_disposable_base_url, str
        ):
            raise TypeError("approved_disposable_base_url must be text")
        if enable_hosted_read_transport and not config.is_hosted_runner_config:
            raise KeystoneOperationNotAllowedError(
                "Hosted read transport requires platform-injected runner configuration."
            )
        if enable_hosted_mutation_transport:
            if not enable_hosted_read_transport or not config.is_hosted_runner_config:
                raise KeystoneOperationNotAllowedError(
                    "Hosted Calendar mutations require the explicitly opted-in hosted transport."
                )
            target = urlsplit(config.base_url)
            hostname = target.hostname or ""
            try:
                ipaddress.ip_address(hostname.rstrip("."))
            except ValueError:
                is_ip_address = False
                try:
                    hostname = (
                        hostname.encode("idna").decode("ascii").rstrip(".").casefold()
                    )
                except UnicodeError:
                    raise KeystoneOperationNotAllowedError(
                        "Disposable Calendar mutation target has an invalid host."
                    ) from None
            else:
                is_ip_address = True
            shared_live_domain = "agentswitch.theschoolofai.in"
            is_shared_live_origin = (
                hostname == shared_live_domain
                or hostname.endswith(f".{shared_live_domain}")
            )
            if target.scheme.casefold() != "https":
                raise KeystoneOperationNotAllowedError(
                    "Disposable Calendar mutations require an HTTPS target."
                )
            if is_shared_live_origin:
                raise KeystoneOperationNotAllowedError(
                    "Calendar mutations are disabled for shared AgentSwitch live origins."
                )
            if is_ip_address:
                raise KeystoneOperationNotAllowedError(
                    "Disposable Calendar mutations require an explicitly named HTTPS host."
                )
            if (
                not approved_disposable_base_url
                or approved_disposable_base_url.rstrip("/") != config.base_url
            ):
                raise KeystoneOperationNotAllowedError(
                    "Disposable Calendar mutations require an exact approved disposable base URL."
                )
        self.config = config
        self._session = session if session is not None else requests.Session()
        self._hosted_read_transport_enabled = enable_hosted_read_transport
        self._hosted_mutation_transport_enabled = enable_hosted_mutation_transport
        self._authorization_identity = object()
        self._discovery_result: ToolDiscoveryResult | None = None
        self._discovery_tools_snapshot: tuple[DiscoveredTool, ...] | None = None
        self._discovery_config: KeystoneConfig | None = None
        self._discovery_session: object | None = None
        self._issued_read_authorizations: WeakSet[KeystoneReadAuthorization] = WeakSet()
        self._consumed_hosted_read_authorizations: WeakSet[KeystoneReadAuthorization] = WeakSet()
        self._issued_mutation_authorizations: WeakSet[KeystoneMutationAuthorization] = WeakSet()
        self._consumed_mutation_authorizations: WeakSet[KeystoneMutationAuthorization] = WeakSet()
        self._state_lock = threading.RLock()
        self._discovery_in_progress = False
        self._calendar_list_attempted = False
        self._next_request_id = 3 if enable_hosted_read_transport else 2
        self._hosted_initialized = False
        self._calendar_list_transport: CalendarListPilotTransport | None = None

    @property
    def hosted_mutation_transport_enabled(self) -> bool:
        return self._hosted_mutation_transport_enabled

    def list_tools(self) -> ToolDiscoveryResult:
        """Initialize opted-in hosted sessions, then fetch the MCP tool catalog."""

        if self._hosted_read_transport_enabled:
            self._initialize_hosted_session()

        with self._state_lock:
            if self._calendar_list_attempted:
                raise KeystoneOperationNotAllowedError(
                    "Tool discovery cannot be refreshed after a Calendar.list pilot attempt."
                )
            if self._discovery_in_progress:
                raise KeystoneOperationNotAllowedError(
                    "Keystone tool discovery is already in progress."
                )
            self._discovery_in_progress = True
            self._clear_discovery_binding()

        discovery: ToolDiscoveryResult | None = None
        try:
            discovery = self._discover_tools()
            return discovery
        finally:
            with self._state_lock:
                if discovery is not None:
                    self._discovery_result = discovery
                    self._discovery_tools_snapshot = deepcopy(discovery.tools)
                    self._discovery_config = self.config
                    self._discovery_session = self._session
                self._discovery_in_progress = False

    def _discover_tools(self) -> ToolDiscoveryResult:
        try:
            token = self.config.require_token()
        except KeystoneConfigurationError:
            raise KeystoneAuthenticationError(
                "Set KEYSTONE_TOKEN to discover Keystone MCP tools."
            ) from None

        request_id = 2 if self._hosted_read_transport_enabled else 1
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {},
        }
        try:
            response = self._session.post(
                self.config.endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=payload,
                timeout=self.config.timeout,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise KeystoneTimeoutError("Keystone tool discovery timed out.") from exc
        except requests.RequestException as exc:
            raise KeystoneTransportError("Keystone tool discovery failed.") from exc

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise KeystoneProtocolError("Keystone tool discovery returned an invalid HTTP status.")
        if not 200 <= status_code < 300:
            if status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected the authenticated tool-discovery request."
                )
            raise KeystoneHttpError(
                f"Keystone returned HTTP {status_code} for tool discovery."
            )

        try:
            message = response.json()
        except ValueError as exc:
            raise KeystoneJsonError("Keystone returned invalid JSON.") from exc

        result = self._parse_response(message, request_id)
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise KeystoneProtocolError("Keystone tools/list result has no tools array.")

        discovered: list[DiscoveredTool] = []
        discovered_names: set[str] = set()
        for item in tools:
            if not isinstance(item, dict):
                raise KeystoneProtocolError("Keystone tools/list contains an invalid tool.")
            name = item.get("name")
            description = item.get("description")
            input_schema = item.get("inputSchema")
            annotations = item.get("annotations", {})
            output_schema = item.get("outputSchema")
            meta = item.get("_meta")
            if (
                not isinstance(name, str)
                or not name
                or name != name.strip()
                or (description is not None and not isinstance(description, str))
                or not isinstance(input_schema, dict)
                or not isinstance(annotations, dict)
            ):
                raise KeystoneProtocolError("Keystone tools/list contains malformed tool metadata.")
            if "outputSchema" in item and not isinstance(output_schema, dict):
                raise KeystoneProtocolError(
                    "Keystone tool outputSchema metadata is malformed."
                )
            if "_meta" in item and not isinstance(meta, dict):
                raise KeystoneProtocolError("Keystone tool _meta metadata is malformed.")
            if name in discovered_names:
                raise KeystoneProtocolError("Keystone tools/list contains duplicate tool names.")
            discovered_names.add(name)
            read_only_hint = annotations.get("readOnlyHint")
            if read_only_hint is not None and type(read_only_hint) is not bool:
                raise KeystoneProtocolError("Keystone tool read-only metadata is malformed.")
            discovered.append(
                DiscoveredTool(
                    name=name,
                    read_only_hint=read_only_hint,
                    description=description,
                    input_schema=dict(input_schema),
                    annotations=dict(annotations),
                    output_schema=(
                        dict(output_schema) if output_schema is not None else None
                    ),
                    meta=dict(meta) if meta is not None else None,
                )
            )

        next_cursor_present = "nextCursor" in result
        next_cursor = result.get("nextCursor")
        if next_cursor_present and next_cursor is not None and (
            not isinstance(next_cursor, str) or not next_cursor.strip()
        ):
            raise KeystoneProtocolError("Keystone tools/list continuation cursor is malformed.")

        collection_evidence = CollectionEvidence(
            (
                PageEvidence(
                    continuation_metadata_present=next_cursor_present,
                    has_more=(
                        next_cursor is not None if next_cursor_present else None
                    ),
                    next_cursor=next_cursor,
                    observed_record_count=len(discovered),
                    record_ids=tuple(tool.name for tool in discovered),
                    truncated=None,
                ),
            ),
            collection_reference="tools/list response",
        )
        completeness = validate_completeness(
            collection_evidence,
            CompletenessContract(
                initial_position_known=True,
                require_truncation_metadata=False,
                max_iterations=1,
            ),
        )
        if completeness.status is not CompletenessStatus.COMPLETE:
            if completeness.status is CompletenessStatus.INCOMPLETE:
                raise KeystoneIncompleteDiscoveryError(
                    completeness, collection_evidence
                )
        discovery = ToolDiscoveryResult(
            tools=tuple(discovered),
            collection_evidence=collection_evidence,
            completeness=completeness,
        )
        return discovery

    def _initialize_hosted_session(self) -> None:
        """Perform the MCP initialization handshake only for hosted opt-in clients."""

        with self._state_lock:
            if self._hosted_initialized:
                return
            config = self.config
            session = self._session
            try:
                token = config.require_token()
            except KeystoneConfigurationError:
                raise KeystoneAuthenticationError(
                    "Set AGENTSWITCH_TOKEN to initialize the hosted Keystone session."
                ) from None

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        initialize_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _HOSTED_MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "eag-v3-calendar-harness",
                    "version": "0.1.0",
                },
            },
        }
        try:
            response = session.post(
                config.endpoint,
                headers=headers,
                json=initialize_payload,
                timeout=config.timeout,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise KeystoneTimeoutError("Keystone MCP initialization timed out.") from exc
        except requests.RequestException as exc:
            raise KeystoneTransportError("Keystone MCP initialization failed.") from exc

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise KeystoneProtocolError(
                "Keystone MCP initialization returned an invalid HTTP status."
            )
        if not 200 <= status_code < 300:
            if status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected MCP initialization."
                )
            raise KeystoneHttpError(
                f"Keystone returned HTTP {status_code} for MCP initialization."
            )
        try:
            message = response.json()
        except ValueError as exc:
            raise KeystoneJsonError(
                "Keystone MCP initialization returned invalid JSON."
            ) from exc

        result = self._parse_response(message, 1)
        if (
            result.get("protocolVersion") != _HOSTED_MCP_PROTOCOL_VERSION
            or not isinstance(result.get("capabilities"), Mapping)
            or not isinstance(result.get("serverInfo"), Mapping)
        ):
            raise KeystoneProtocolError(
                "Keystone MCP initialization result does not match the supported protocol."
            )

        try:
            response = session.post(
                config.endpoint,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
                timeout=config.timeout,
                allow_redirects=False,
                # The server answers 202 but never finishes the body, so do not wait for it.
                stream=True,
            )
        except requests.Timeout as exc:
            raise KeystoneTimeoutError(
                "Keystone MCP initialized notification timed out."
            ) from exc
        except requests.RequestException as exc:
            raise KeystoneTransportError(
                "Keystone MCP initialized notification failed."
            ) from exc

        status_code = getattr(response, "status_code", None)
        close = getattr(response, "close", None)
        if callable(close):
            close()
        if type(status_code) is not int:
            raise KeystoneProtocolError(
                "Keystone MCP initialized notification returned an invalid HTTP status."
            )
        if not 200 <= status_code < 300:
            if status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected the MCP initialized notification."
                )
            raise KeystoneHttpError(
                f"Keystone returned HTTP {status_code} for MCP initialized notification."
            )
        with self._state_lock:
            if self.config is not config or self._session is not session:
                raise KeystoneOperationNotAllowedError(
                    "Hosted Keystone client state changed during MCP initialization."
                )
            self._hosted_initialized = True

    def authorize_read_request(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> KeystoneReadAuthorization:
        """Bind a local policy decision to this client's observed catalog/session."""

        with self._state_lock:
            discovery = self._require_owned_discovery()
            from .authorization import validate_resource_read_policy

            descriptor = validate_resource_read_policy(discovery, operation, arguments)
            try:
                arguments_json = json.dumps(
                    dict(arguments),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError):
                raise KeystoneProtocolError(
                    "Authorized arguments cannot be represented as JSON."
                ) from None
            authorization = KeystoneReadAuthorization(
                operation=operation,
                _arguments_json=arguments_json,
                _client_identity=self._authorization_identity,
                _config=self.config,
                _session=self._session,
                _discovery=discovery,
                _descriptor_snapshot=deepcopy(descriptor),
            )
            self._issued_read_authorizations.add(authorization)
            return authorization

    def validate_read_authorization(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> DiscoveredTool:
        """Recheck a client-issued decision without invoking transport."""

        with self._state_lock:
            if (
                not isinstance(authorization, KeystoneReadAuthorization)
                or authorization not in self._issued_read_authorizations
                or authorization._client_identity is not self._authorization_identity
                or authorization._config is not self.config
                or authorization._session is not self._session
            ):
                raise KeystoneOperationNotAllowedError(
                    "Read authorization is not bound to this Keystone client."
                )
            discovery = self._require_owned_discovery()
            if authorization._discovery is not discovery:
                raise KeystoneOperationNotAllowedError(
                    "Read authorization belongs to a different discovery response."
                )
            from .authorization import validate_resource_read_policy

            descriptor = validate_resource_read_policy(
                discovery,
                authorization.operation,
                authorization.arguments,
            )
            if descriptor != authorization._descriptor_snapshot:
                raise KeystoneOperationNotAllowedError(
                    "The discovered descriptor changed after authorization."
                )
            return descriptor

    def authorize_mutation_request(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> KeystoneMutationAuthorization:
        """Bind one allowed Calendar mutation to hosted opt-in and discovery."""

        if not self._hosted_mutation_transport_enabled:
                raise KeystoneOperationNotAllowedError(
                    "Disposable hosted Calendar mutation transport is disabled."
                )
        with self._state_lock:
                discovery = self._require_owned_discovery()
                from .authorization import validate_disposable_calendar_mutation_policy

                descriptor = validate_disposable_calendar_mutation_policy(
                    discovery, operation, arguments
                )
                try:
                    arguments_json = json.dumps(
                        dict(arguments),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                except (TypeError, ValueError):
                    raise KeystoneProtocolError(
                        "Calendar mutation arguments cannot be represented as JSON."
                    ) from None
                authorization = KeystoneMutationAuthorization(
                    operation=operation,
                    _arguments_json=arguments_json,
                    _client_identity=self._authorization_identity,
                    _config=self.config,
                    _session=self._session,
                    _discovery=discovery,
                    _descriptor_snapshot=deepcopy(descriptor),
                )
                self._issued_mutation_authorizations.add(authorization)
                return authorization

    def validate_mutation_authorization(
        self,
        authorization: KeystoneMutationAuthorization,
    ) -> DiscoveredTool:
        """Revalidate the one-use disposable mutation authorization."""

        with self._state_lock:
                if (
                    not self._hosted_mutation_transport_enabled
                    or not isinstance(authorization, KeystoneMutationAuthorization)
                    or authorization not in self._issued_mutation_authorizations
                    or authorization._client_identity is not self._authorization_identity
                    or authorization._config is not self.config
                    or authorization._session is not self._session
                ):
                    raise KeystoneOperationNotAllowedError(
                        "Mutation authorization is not enabled or bound to this Keystone client."
                    )
                discovery = self._require_owned_discovery()
                if authorization._discovery is not discovery:
                    raise KeystoneOperationNotAllowedError(
                        "Mutation authorization belongs to a different discovery response."
                    )
                from .authorization import validate_disposable_calendar_mutation_policy

                descriptor = validate_disposable_calendar_mutation_policy(
                    discovery,
                    authorization.operation,
                    authorization.arguments,
                )
                if descriptor != authorization._descriptor_snapshot:
                    raise KeystoneOperationNotAllowedError(
                        "The mutation descriptor changed after authorization."
                    )
                return descriptor

    def _clear_discovery_binding(self) -> None:
        self._discovery_result = None
        self._discovery_tools_snapshot = None
        self._discovery_config = None
        self._discovery_session = None
        self._issued_read_authorizations.clear()
        self._consumed_hosted_read_authorizations.clear()
        self._issued_mutation_authorizations.clear()
        self._consumed_mutation_authorizations.clear()

    def _require_owned_discovery(self) -> ToolDiscoveryResult:
        if self._discovery_in_progress:
            raise KeystoneOperationNotAllowedError(
                "Keystone discovery is being refreshed."
            )
        discovery = self._discovery_result
        if discovery is None or self._discovery_tools_snapshot is None:
            raise KeystoneOperationNotAllowedError(
                "This Keystone client has no successful tools/list discovery."
            )
        if self.config is not self._discovery_config or self._session is not self._discovery_session:
            raise KeystoneOperationNotAllowedError(
                "Client configuration or session changed after discovery."
            )
        if discovery.tools != self._discovery_tools_snapshot:
            raise KeystoneOperationNotAllowedError(
                "Discovered tool metadata changed after discovery."
            )
        return discovery

    def call_read(
        self,
        operation: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reject raw tool requests; callers must use a client-issued authorization."""

        raise KeystoneOperationNotAllowedError(
            "No Keystone operation has trusted approval; transport is disabled."
        )

    def call_authorized_read(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> dict[str, Any]:
        """Perform one schema-validated hosted read from the exact resource allowlist."""

        if not self._hosted_read_transport_enabled:
            raise KeystoneOperationNotAllowedError(
                "Keystone read transport is disabled."
            )

        from .authorization import RESOURCE_READ_OPERATIONS
        from .mcp_response import parse_mcp_tool_call_response

        with self._state_lock:
            descriptor = self.validate_read_authorization(authorization)
            if authorization.operation not in RESOURCE_READ_OPERATIONS:
                raise KeystoneOperationNotAllowedError(
                    "Keystone hosted read is not in the exact resource allowlist."
                )
            if authorization in self._consumed_hosted_read_authorizations:
                raise KeystoneOperationNotAllowedError(
                    "A Keystone hosted read authorization can be used only once."
                )
            if not isinstance(descriptor.output_schema, Mapping):
                raise KeystoneProtocolError(
                    "Keystone hosted reads require a discovered output schema."
                )
            config = self.config
            session = self._session
            try:
                token = config.require_token()
            except KeystoneConfigurationError:
                raise KeystoneAuthenticationError(
                    "Set AGENTSWITCH_TOKEN for hosted Keystone reads."
                ) from None
            request_id = self._next_request_id
            self._next_request_id += 1
            self._consumed_hosted_read_authorizations.add(authorization)

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": authorization.operation,
                "arguments": authorization.arguments,
            },
        }
        try:
            response = session.post(
                config.endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=payload,
                timeout=config.timeout,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise KeystoneTimeoutError("Keystone hosted read timed out.") from exc
        except requests.RequestException as exc:
            raise KeystoneTransportError("Keystone hosted read failed.") from exc

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise KeystoneProtocolError("Keystone hosted read returned an invalid HTTP status.")
        if not 200 <= status_code < 300:
            if status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected the authenticated hosted read."
                )
            raise KeystoneHttpError(
                f"Keystone returned HTTP {status_code} for a hosted read."
            )
        try:
            message = response.json()
        except ValueError as exc:
            raise KeystoneJsonError("Keystone hosted read returned invalid JSON.") from exc

        parsed = parse_mcp_tool_call_response(
            message,
            request_id,
            descriptor.output_schema,
            expose_structured_content=True,
        )
        if (
            not parsed.pagination_ready
            or parsed.structured_content is None
            or not parsed.output_schema_validated
        ):
            raise KeystoneProtocolError(
                "Keystone hosted read did not return a validated structured resource page."
            )
        return dict(parsed.structured_content)

    def call_authorized_mutation(
        self,
        authorization: KeystoneMutationAuthorization,
    ) -> dict[str, Any]:
        """Perform one explicitly authorized event create/update on a disposable copy."""

        if not self._hosted_mutation_transport_enabled:
            raise KeystoneOperationNotAllowedError(
                "Disposable hosted Calendar mutation transport is disabled."
            )

        from .mcp_response import parse_mcp_tool_call_response

        with self._state_lock:
            descriptor = self.validate_mutation_authorization(authorization)
            if authorization in self._consumed_mutation_authorizations:
                raise KeystoneOperationNotAllowedError(
                    "A disposable Calendar mutation authorization can be used only once."
                )
            if not isinstance(descriptor.output_schema, Mapping):
                raise KeystoneProtocolError(
                    "Calendar mutations require a discovered output schema."
                )
            config = self.config
            session = self._session
            try:
                token = config.require_token()
            except KeystoneConfigurationError:
                raise KeystoneAuthenticationError(
                    "Set AGENTSWITCH_TOKEN for the disposable Calendar mutation."
                ) from None
            request_id = self._next_request_id
            self._next_request_id += 1
            self._consumed_mutation_authorizations.add(authorization)

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": authorization.operation,
                "arguments": authorization.arguments,
            },
        }
        try:
            response = session.post(
                config.endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=payload,
                timeout=config.timeout,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise KeystoneTimeoutError(
                "Disposable Calendar mutation timed out; verify state before retrying."
            ) from exc
        except requests.RequestException as exc:
            raise KeystoneTransportError(
                "Disposable Calendar mutation transport failed; verify state before retrying."
            ) from exc

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise KeystoneProtocolError(
                "Disposable Calendar mutation returned an invalid HTTP status."
            )
        if not 200 <= status_code < 300:
            if status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected the disposable Calendar mutation."
                )
            raise KeystoneHttpError(
                f"Keystone returned HTTP {status_code} for a Calendar mutation."
            )
        try:
            message = response.json()
        except ValueError as exc:
            raise KeystoneJsonError(
                "Disposable Calendar mutation returned invalid JSON."
            ) from exc

        parsed = parse_mcp_tool_call_response(
            message,
            request_id,
            descriptor.output_schema,
            expose_structured_content=True,
        )
        if (
            parsed.structured_content is None
            or not parsed.output_schema_validated
            or not isinstance(parsed.structured_content.get("id"), str)
        ):
            raise KeystoneProtocolError(
                "Calendar mutation returned no schema-valid event ID; re-read state before retrying."
            )
        return dict(parsed.structured_content)

    def _call_calendar_list_pilot(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> ParsedMcpToolCallResponse:
        """Dispatch only through this client's explicitly configured pilot transport."""

        from .calendar_list_transport import CalendarListPilotTransport

        with self._state_lock:
            transport = self._calendar_list_transport
            if type(transport) is not CalendarListPilotTransport:
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport is not configured."
                )
            if not transport._is_bound_to(self, self.config):
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport is not bound to this client."
                )
        return transport.send_once(authorization)

    def _claim_calendar_list_pilot_request(
        self,
        authorization: KeystoneReadAuthorization,
        transport: CalendarListPilotTransport,
    ) -> _CalendarListPilotRequest:
        """Validate and consume client state before handing a request to its transport."""

        with self._state_lock:
            config = self.config
            session = self._session
            discovery = self._discovery_result
            if (
                transport is not self._calendar_list_transport
                or not transport._is_bound_to(self, config)
            ):
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport is not bound to this client."
                )

            descriptor = self.validate_read_authorization(authorization)
            arguments = authorization.arguments
            if (
                self.config is not config
                or self._session is not session
                or self._discovery_result is not discovery
            ):
                raise KeystoneOperationNotAllowedError(
                    "Client state changed during Calendar.list authorization."
                )
            descriptor_snapshot = deepcopy(descriptor)
            authorized_descriptor = deepcopy(authorization._descriptor_snapshot)
            if descriptor_snapshot != authorized_descriptor:
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list descriptor changed during authorization."
                )
            if (
                authorization.operation != "Calendar.list"
                or set(arguments) != {"limit", "offset"}
                or type(arguments.get("limit")) is not int
                or arguments["limit"] != 1
                or type(arguments.get("offset")) is not int
                or arguments["offset"] != 0
                or descriptor_snapshot.name != "Calendar.list"
            ):
                raise KeystoneOperationNotAllowedError(
                    "Only the fixed Calendar.list pilot request is permitted."
                )
            if self._calendar_list_attempted:
                raise KeystoneOperationNotAllowedError(
                    "The Calendar.list pilot attempt has already been consumed."
                )
            try:
                config.require_token()
            except KeystoneConfigurationError:
                raise KeystoneAuthenticationError(
                    "Set KEYSTONE_TOKEN to invoke the approved Calendar.list pilot."
                ) from None

            request_id = self._next_request_id
            self._next_request_id += 1
            self._calendar_list_attempted = True
            return _CalendarListPilotRequest(
                _client_identity=self._authorization_identity,
                _transport_identity=transport._identity,
                endpoint=config.endpoint,
                timeout=config.timeout,
                request_id=request_id,
                output_schema=deepcopy(descriptor_snapshot.output_schema),
            )

    @staticmethod
    def _parse_response(message: Any, request_id: int) -> dict[str, Any]:
        """Validate a JSON-RPC result without exposing server-provided diagnostics."""

        if type(request_id) is not int:
            raise KeystoneProtocolError("Keystone request ID is malformed.")
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise KeystoneProtocolError("Keystone response is not a JSON-RPC 2.0 object.")
        if type(message.get("id")) is not int or message["id"] != request_id:
            raise KeystoneProtocolError("Keystone response ID does not match the request.")
        if "error" in message:
            if not isinstance(message["error"], dict):
                raise KeystoneProtocolError("Keystone JSON-RPC error is malformed.")
            raise KeystoneRpcError("Keystone returned a JSON-RPC error.")
        result = message.get("result")
        if not isinstance(result, dict):
            raise KeystoneProtocolError("Keystone JSON-RPC result is missing or malformed.")
        if result.get("isError") is True:
            raise KeystoneToolError("Keystone MCP operation reported an error.")
        return result