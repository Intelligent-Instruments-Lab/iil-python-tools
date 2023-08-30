import sys
try:
    from rich.panel import Panel
    from rich.pretty import Pretty

    from textual.app import App, ComposeResult
    from textual.reactive import reactive
    from textual.widgets import Header, Footer, Static, Button, TextLog, Label
    from textual.css.query import NoMatches, TooManyMatches

    class TUI(App):
        """Base Textual app for iipyper programs."""
        ## e.g.
        # CSS_PATH = 'my_tui.css'
        # BINDINGS = [
        #     ("a", "my_a_action", "Do something when A is pressed"),
        #     ("b", "my_b_action", "Do somethign when B is pressed")]

        def __init__(self):
            super().__init__()
            self.std_log = TextLog(id='std_log')

        def compose(self) -> ComposeResult:
            """Create child widgets for the Textual App.
            
            override this to build the TUI for your iipyper app.
            """
            yield Header()
            yield self.std_log
            yield Footer()

        def set_action(self, f):
            """
            Decorator attaching a function to a Textual Button or key bind.
            
            the name of the decorated function should match the id of a child Button node and/or a key binding in the TUI.
            """
            setattr(self, f'action_{f.__name__}', f)
            return f
        
        def on(self, f):
            """
            Decorator attaching a function to a Textual event

            The name of the decorated function should be an event such as:
                mount
                mouse_move
                click
            """
            if f.__name__=='mount':
                self._mount = f
            else:
                # this doesn't work for on_mount for whatever reason
                setattr(self, f'on_{f.__name__}', f)
            return f
        
        def on_mount(self):
            # self.std_log.write('MOUNT')
            try:
                self._mount()
            except Exception as e:
                print(e)
                pass
        
        def set_mouse_move(self, f):
            self.on_mouse_move = f
            return f

        def on_button_pressed(self, event: Button.Pressed) -> None:
            getattr(self, f'action_{event.button.id}')()


        # def _print(self, k, *v):
        #     for s in (k, *v):
        #         self.std_log.write(s)
        #    # self.std_log.write(' '.join(str(s) for s in (k, *v)))

        def flush(self): pass
        def write(self, s):
            """for redirecting stdout to a file-like"""
            if self.is_running:
                return self.call_from_anywhere(self.std_log.write, s)
            else:
                # TODO: buffer and write after start
                return sys.__stdout__.write(s)

        def print(self, *a, **kw):
            """redirects to the UI's default std output log"""
            kw['file'] = self
            print(*a, **kw)
            # if self.is_running:
            #     self.call_from_anywhere(self._print, *a, **kw)
            #     # try:
            #     #     self.call_from_thread(self._print, *a, **kw)
            #     # except:
            #     #     self._print(*a, **kw)
            # else:
            #     print(*a, **kw)

        def call_from_anywhere(self, f, *a, **kw):
            try:
                return self.call_from_thread(f, *a, **kw)
            except RuntimeError as e:
                return f(*a, **kw)

        def __call__(self, *a, **kw):
            if self.is_running:
                self.call_from_anywhere(self._call, *a, **kw)
                # try:
                #     self.call_from_thread(self._call, *a, **kw)
                # except:
                #     self.do_call(*a, **kw)
            else:
                print(*a)
                for k,v in kw.items():
                    print(k, '->', v)

        def _call(self, **kw):
            """
            by default, expects your TUI to have child nodes
            with a reactive `value` attribute, which updates the node when set.
            then calling my_tui(my_node_id=my_value) will update the node.
            """
            if not self.is_running:
                return
            for k in kw:
                try:
                    node = self.query_one('#'+k)
                    node.value = kw[k]
                except (NoMatches, TooManyMatches):
                    self.print(f'TUI: node "{k}" not found')
                except AttributeError:
                    self.print(f'TUI: node "{k}" lacks value "reactive"')


except ImportError as e:
    print(e.msg)
    print('install package `textual` for terminal user interface')

    class TUI:
        def print(self, *a, **kw):
            print(*a, **kw)
        def __call__(self, *a, **kw):
            pass
        def set_action(self, f):
            pass
        def run(self):
            pass
