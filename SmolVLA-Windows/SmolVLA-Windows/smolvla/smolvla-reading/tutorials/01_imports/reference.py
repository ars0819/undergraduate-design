"""第 1 课参考答案。"""

from task_labels import format_episode


def main():
    task_id = 2
    episode_number = 3
    label = format_episode(task_id, episode_number)
    print(label)


if __name__ == "__main__":
    main()
