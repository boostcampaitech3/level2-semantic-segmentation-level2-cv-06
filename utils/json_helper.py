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
        self.__image_annotations = defaultdict(list)
        self.__image_annotations_by_categories = {}
        for image in self.__images:
            self.__image_annotations_by_categories.update({image["id"]: defaultdict(list)})
        for annotation in self.__annotations:
            img_id = annotation["image_id"]
            cat_id = annotation["category_id"]
            anno_id = annotation["id"]
            self.__image_annotations_by_categories[img_id][cat_id].append(anno_id)
            self.__image_annotations[img_id].append(anno_id)

        print("finished initializing json helper")
        
    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, idx):
        return self.__images[idx], self.__annotations[idx]

    def get_image_list(self) -> list:
        return [{"id": img["id"], "file_name": img["file_name"]} for img in self.__images]

    def showAnns(self, anns, draw_bbox=False):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon
        import numpy as np

        ax = plt.gca()  # TODO should search what this is
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            # segmentation
            # polygon
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((int(len(seg)/2), 2))
                polygons.append(Polygon(poly))
                color.append(c)
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

    def get_annotations(self, img_id: int, category_id=None) -> list:
        """
        Return
        ---
        list of annotations
        """
        if not category_id:
            anno_ids = set(self.__image_annotations[img_id])
        else:
            anno_ids = set(self.__image_annotations_by_categories[img_id][category_id])
        return [anno for anno in self.__annotations if anno["id"] in anno_ids]
            
        
            
if __name__ == '__main__':
    # test code
    classes = ["Gene"]
    jh = JsonHelper("C:/Users/alsrl/.ssh/temp.json")
    print(f"len(jh): {len(jh)}")
    print(jh.__image_annotations[0][8])
