from calendar_agent.models import AgentTask


def task() -> AgentTask:
    return AgentTask(
        goal="calendar.find_30_minutes",
        instruction="Find 30 minutes with the plant head",
    )