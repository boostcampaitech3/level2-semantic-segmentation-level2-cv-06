import json
from collections import defaultdict


class JsonHelper:
    def __init__(self, json_path:str, classes: list=None) -> None:
        print("initializing json helper")
        self.__json_path = json_path
        self.__classes = classes

        print(f"loading json file, path: {self.__json_path}")
        with open(json_path, "r") as file:
            self.__data = json.load(file)
        self.__columns = self.__data.keys()
        self.__images = self.__data["images"]
        self.__annotations = self.__data["annotations"]
        if not self.__classes:
            self.__classes = []
            for category in self.__data["categories"]:
                self.__classes.append((category["id"], category["name"]))
            self.__classes.sort(key=lambda x:x[0])
            self.__classes = [class_name for class_id, class_name in self.__classes]

        print("mapping images information to annotations")
        self.__image_annotations = {}
        for image in self.__images:
            self.__image_annotations.update({image["id"]: defaultdict(list)})
        for annotation in self.__annotations:
            img_id = annotation["image_id"]
            cat_id = annotation["category_id"]
            anno_id = annotation["id"]
            self.__image_annotations[img_id][cat_id].append(anno_id)

        print("finished initializing json helper")
        
    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, idx):
        return self.__images[idx], self.__annotations[idx]

    def get_image_list(self) -> list:
        return [(img["id"], img["file_name"]) for img in self.__images]
            

if __name__ == '__main__':
    # test code
    classes = ["Gene"]
    jh = JsonHelper("C:/Users/alsrl/.ssh/temp.json")
    print(f"len(jh): {len(jh)}")
    print(jh.__image_annotations[0][8])
