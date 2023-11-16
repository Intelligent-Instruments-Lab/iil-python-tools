import taichi as ti
import time

class Taichi:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.gpu      = kwargs.get('gpu', 'vulkan')
        self.cpu      = kwargs.get('cpu', None)
        self.fps      = kwargs.get('fps', 120)
        self.seed     = kwargs.get('seed', int(time.time()))
        self.headless = kwargs.get('headless', False)
        self.init()
    def init(self):
        self.init_ti()
        self.init_ui()
        print(f"[Tölvera] Taichi initialised with: {vars(self)}")
    def init_ti(self):
        if self.cpu:
            ti.init(arch=ti.cpu, random_seed=self.seed)
            self.gpu = None
            print("[Tölvera] Running on CPU")
        else:
            if self.gpu == "vulkan":
                ti.init(arch=ti.vulkan, random_seed=self.seed)
            elif self.gpu == "cuda":
                ti.init(arch=ti.cuda, random_seed=self.seed)
            else:
                print(f"[Tölvera] Invalid GPU: {self.gpu}")
                return False
            print(f"[Tölvera] Running on {self.gpu}")
    def init_ui(self):
        self.window = ti.ui.Window(self.tv.name, (self.tv.x, self.tv.y), fps_limit=self.fps, show_window=not self.headless)
        self.canvas = self.window.get_canvas()
    def show(self, px):
        self.canvas.set_image(px.px.rgba)
        if not self.headless: self.window.show()
