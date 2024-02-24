import os
from operator import itemgetter
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty

class LoadFolderDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def sort_by_date(self, files, filesystem):
        result = sorted(
            zip(
                files,
                list(map(os.path.isdir, files)),
                list(map(os.path.getmtime, files))
            ),
            key=itemgetter(1,2),
            reverse=True)
        return list(map(itemgetter(0),result))

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def sort_by_date(self, files, filesystem):
        result = sorted(
            zip(
                files,
                list(map(os.path.isdir, files)),
                list(map(os.path.getmtime, files))
            ),
            key=itemgetter(1,2),
            reverse=True)
        return list(map(itemgetter(0),result))

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def sort_by_date(self, files, filesystem):
        result = sorted(
            zip(
                files,
                list(map(os.path.isdir, files)),
                list(map(os.path.getmtime, files))
            ),
            key=itemgetter(1,2),
            reverse=True)
        return list(map(itemgetter(0),result))
