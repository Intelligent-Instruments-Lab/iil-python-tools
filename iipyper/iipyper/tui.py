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

        def on_button_pressed(self, event: Button.Pressed) -> None:
            getattr(self, f'action_{event.button.id}')()

        def _print(self, k, *v):
            self.std_log.write(' '.join(str(s) for s in (k, *v)))

        def print(self, *a, **kw):
            """redirects to the TUI's default std output log"""
            if self.is_running:
                try:
                    self.call_from_thread(self._print, *a, **kw)
                except:
                    self._print(*a, **kw)

        def __call__(self, *a, **kw):
            if self.is_running:
                try:
                    self.call_from_thread(self.do_call, *a, **kw)
                except:
                    self.do_call(*a, **kw)
            else:
                print(*a)
                for k,v in kw.items():
                    print(k, '->', v)

        def do_call(self, **kw):
            """
            Optionally override this as the main output for your iipyper app.

            by default, it will expect your TUI to have child nodes
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
                    self.print(f'TUI: node "{k}" lacks `value` reactive')
            # raise NotImplementedError
            ## e.g.
            # if my_keyword_arg is not None:
                # self.my_node.write(my_keyword_arg)


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
