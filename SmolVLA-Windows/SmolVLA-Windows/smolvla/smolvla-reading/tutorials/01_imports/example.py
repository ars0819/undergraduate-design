"""从另一个文件导入函数，再用它打印两个任务名称。"""

from task_labels import format_task


def print_plan(task_ids):
    for task_id in task_ids:
        label = format_task(task_id)
        print(label)


def main():
    task_ids = [0, 1]
    print_plan(task_ids)


if __name__ == "__main__":
    main()
