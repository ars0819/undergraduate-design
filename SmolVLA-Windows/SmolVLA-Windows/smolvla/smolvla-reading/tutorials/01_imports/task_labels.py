"""根据编号生成任务标签和回合标签。"""


def format_task(task_id):
    return f"任务 {task_id}"


def format_episode(task_id, episode_number):
    return f"{format_task(task_id)}，第 {episode_number} 回合"
