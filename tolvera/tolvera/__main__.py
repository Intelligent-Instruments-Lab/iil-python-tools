'''
TODO: 'sketchbook' run/export scripts?
'''

import fire

from .__init__ import Tolvera

def help():
    print('')

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render
    def _():
        tv.p()

if __name__=='__main__':
    fire.Fire(main)
