from kivy.lang import Builder
from kivy.factory import Factory
from kivy.core.window import Window
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.uix.modalview import ModalView
from kivymd.app import MDApp
from kivymd.toast import toast
from kivymd.uix.filemanager import MDFileManager
from kivy.core.audio import SoundLoader


Builder.load_string(
    """
<FileManager@BoxLayout>:
    orientation: "vertical"
    spacing: dp(5)

    MDToolbar:
        id: toolbar
        title: app.title
        left_action_items: [["menu", lambda x: None]]
        elevation: 10
        md_bg_color: app.theme_cls.primary_color


    FloatLayout:

        MDRoundFlatIconButton:
            text: "Load your file:"
            icon: "folder"
            pos_hint: {"center_x": .5, "center_y": .6}
            on_release: app.file_manager_open()
            
            
         
            
        
"""
)


class MainApp(MDApp):
    manager_open = BooleanProperty()
    manager = ObjectProperty()
    file_manager = ObjectProperty()
    sound = SoundLoader.load('id.wav')

    def __init__(self, **kwargs):
        self.title = "Music app"
        self.theme_cls.primary_palette = "Teal"
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)

    def build(self):
        self.root = Factory.FileManager()

    def file_manager_open(self):
        if not self.manager:
            self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
            self.file_manager = MDFileManager(
                exit_manager=self.exit_manager, select_path=self.select_path
            )
            self.manager.add_widget(self.file_manager)
            self.file_manager.show("/")
        self.manager_open = True
        self.manager.open()

    def select_path(self, path):

        self.exit_manager()
        toast(path)
        sound = SoundLoader.load(path)
        sound.play()


    def exit_manager(self, *args):


        self.manager.dismiss()
        self.manager_open = False

    def events(self,keyboard ):


        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True


if __name__ == "__main__":
    MainApp().run()