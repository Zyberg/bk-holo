from .image import Image
import os

class StateManager:
    def __init__(self):
        self.paths = []
        self.images = {}

        self.reference = None
        self.object = None

    # TODO: refactor to use path
    def set_reference(self, path_index):
        if path_index not in self.images:
            self.load_images(path_index)
        
        self.reference = self.images[path_index]

    def set_object(self, path_index):
        if path_index not in self.images:
            self.load_images(path_index)
        
        self.object = self.images[path_index]

    def load_directory(self, directory):
        script_dir = os.path.dirname(__file__)
        directory_path = os.path.join(script_dir, directory)

        for filename in os.listdir(directory_path):
            if filename.endswith('.tif'):
                file_path = os.path.join(directory_path, filename)
                self.paths.append(file_path)

    def load_images(self, path_index = None):
        if path_index != None: 
            self.__load_image(self.paths[path_index], path_index)
        else:
            for path_index, path in enumerate(self.paths):
                self.__load_image(path, path_index)

    def __load_image(self, path, index):
        self.images[index] = Image(path)
