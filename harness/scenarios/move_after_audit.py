from calendar_agent.models import AgentTask


def task() -> AgentTask:
    return AgentTask(
        goal="calendar.move_after_audit",
        instruction="Move everything after the audit",
    )