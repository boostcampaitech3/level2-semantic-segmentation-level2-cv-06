import json
from collections import defaultdict


class JsonHelper:
    def __init__(self, json_path:str, classes: list=None) -> None:
        self.json_path = json_path
        self.classes = classes

        print(f"loading json file, path: {self.json_path}")
        with open(json_path, "r") as file:
            self.data = json.load(file)
        self.__columns = self.data.keys()
        self.images = self.data["images"]
        self.annotations = self.data["annotations"]
        if not self.classes:
            self.classes = []
            for category in self.data["categories"]:
                self.classes.append((category["id"], category["name"]))
            self.classes.sort(key=lambda x:x[0])
            self.classes = [class_name for class_id, class_name in self.classes]

        print("mapping images information to annotations")
        self.image_annotations = {}
        for image in self.images:
            self.image_annotations.update({image["id"]: defaultdict(list)})
        for annotation in self.annotations:
            img_id = annotation["image_id"]
            cat_id = annotation["category_id"]
            anno_id = annotation["id"]
            self.image_annotations[img_id][cat_id].append(anno_id)
        
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.annotations[idx]


if __name__ == '__main__':
    # test code
    classes = ["Gene"]
    jh = JsonHelper("C:/Users/alsrl/.ssh/temp.json")
    print(f"len(jh): {len(jh)}")
    print(jh.image_annotations[0][8])
