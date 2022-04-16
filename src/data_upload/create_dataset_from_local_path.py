from clearml import Dataset,Task
import argparse

OUTPUT_URL = "s3://derekchiaxyz/storage"
OUTPUT_URL = "s3://experiment-logging/storage"


def create_dataset(folder_path, dataset_project, dataset_name):
    """
    Checks if parent dataset exists
    - if yes, finalise the parent dataset, then create new child dataset and point to parent
    - if no, create new dataset as parent
    """
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        parent_dataset.finalize()
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        # child_dataset.add_files(folder_path)
        child_dataset.sync_folder(folder_path)
        child_dataset.upload()
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        dataset.add_files(folder_path)
        dataset.sync_folder(folder_path)
        dataset.upload()
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    """Get last dataset child object"""
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])


def create_dataset_and_finalize(folder_path, dataset_project, dataset_name):
    """
    Create ClearML Dataset and finalise the object
    Note: If there is an existing parent dataset, this will throw an error. Consider using create_dataset() instead
    """
    dataset = Dataset.create(dataset_name, dataset_project)
    dataset.sync_folder(folder_path)
    dataset.upload(output_url=OUTPUT_URL)
    dataset.finalize()
    return True


if __name__ == "__main__":
    """
    Example usage:

    > python create_dataset_from_local_path.py --folder_path "/home/derek/Desktop/project-m/data/maritime/fleetmon-news/details-small" --dataset_project "project-m/dataset" --dataset_name "maritime_fleetmon_news_html_small"

    > python create_dataset_from_local_path.py --folder_path "/home/derek/Desktop/project-m/data/maritime/fleetmon-news/details" --dataset_project "project-m/dataset" --dataset_name "maritime_fleetmon_news_html_all"
    """
    # parser = argparse.ArgumentParser(description="Create dataset from local path")
    # parser.add_argument("--folder_path", type=str, required=False)
    # parser.add_argument("--dataset_project", type=str, required=True)
    # parser.add_argument("--dataset_name", type=str, required=True)
    # args = parser.parse_args()

    # result = create_dataset_and_finalize(
    #     args.folder_path, args.dataset_project, args.dataset_name
    # )
    # if result:
    #     print(
    #         "{} - {} created with files from {}".format(
    #             args.folder_path, args.dataset_project, args.dataset_name
    #         )
    #     )
    # else:
    #     print("Error!")

    task = Task.init(project_name="BLINK", task_name="upload embeddings file")
    dataset = create_dataset(
        folder_path="models",
        dataset_project="BLINK/dataset",
        dataset_name="BLINK_models",
    )
    dataset.finalize()
