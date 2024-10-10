import torch
import matplotlib.patches as patches

class Point:
    """
    A class representing a 2D point.
    """
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class BoundingBox:
    """
    A class representing a bounding box.
    """
    def __init__(self, x: float, y: float, width: float, height: float, normalized: bool = True):
        self.x: float = x
        self.y: float = y
        self.width: float = width
        self.height: float = height
        self.left = x - width / 2
        self.top: float = y - height / 2
        self.normalized = normalized

    def print(self):
        """
        Prints the bounding box coordinates and dimensions.
        """
        print(f"left: {self.left}, top: {self.top}, width: {self.width}, height: {self.height}")

    def area(self) -> int:
        """
        Calculates the area of the bounding box.
        Returns:
            (int): The area of the bounding box.
        """
        return self.width * self.height

    def translation_from_center(self, new_center: Point) -> 'BoundingBox':
        """
        Returns a new BoundingBox, translated from the current BoundingBox, with the new center.
        Args:
            new_center (Point): The new center.
        Returns:
            (BoundingBox): A new BoundingBox, translated from the current BoundingBox, with the new center.
        """
        dx = new_center.x - self.center.x
        dy = new_center.y - self.center.y
        return BoundingBox(self.left + dx, self.top + dy, self.width, self.height)

    @staticmethod
    def intersection_area(bbox_1: 'BoundingBox', bbox_2: 'BoundingBox') -> int:
        """
        Calculates the intersection area of two bounding boxes.
        Args:
            bbox_1 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
            bbox_2 (BoundingBox): Object representing the bounding box, with attributes left, top, width, and height.
        Returns:
            (int): The intersection area of the two bounding boxes.
        """
        xA, yA = max(bbox_1.left, bbox_2.left), max(bbox_1.top, bbox_2.top)
        xB, yB = min(bbox_1.left + bbox_1.width, bbox_2.left + bbox_2.width), min(bbox_1.top + bbox_1.height, bbox_2.top + bbox_2.height)

        if xB <= xA or yB <= yA:
            return 0
        
        return (xB - xA) * (yB - yA)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'BoundingBox':
        """
        Creates a BoundingBox object from a tensor.
        Args:
            tensor (torch.Tensor): A tensor representing a bounding box, with shape (4,).
        Returns:
            (BoundingBox): A BoundingBox object created from the tensor.
        """
        return BoundingBox(int(tensor[0]), int(tensor[1]), int(tensor[2]), int(tensor[3]))

    def denormalized(self, image_width: int, image_height: int) -> 'BoundingBox':
        """
        Denormalizes the bounding box coordinates.
        Args:
            image_width (int): The width of the image.
            image_height (int): The height of the image.
        Returns:
            (BoundingBox): The denormalized bounding box.
        """
        return BoundingBox(
            x=self.x * image_width,
            y=self.y * image_height,
            width=self.width * image_width,
            height=self.height * image_height,
            normalized=False
        )

    def plt_rectangle(self, image_width: int, image_height: int) -> patches.Rectangle:
        """
        Return a matplotlib Rectangle object representing the bounding box.

        Args:
            image_width (int): The width of the image.
            image_height (int): The height of the image
        Returns:
            patches.Rectangle: A matplotlib Rectangle object representing the bounding box.
        """
        bbox = self
        if self.normalized:
            bbox = self.denormalized(image_width, image_height)
    
        return patches.Rectangle(
            (bbox.left, bbox.top),
            bbox.width,
            bbox.height,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )