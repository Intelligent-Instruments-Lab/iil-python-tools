'''
TODO: 'sketchbook' run/export scripts?
'''

import fire

from tolvera import Tolvera

def help():
    print('')

def main(**kwargs):
    tv = Tolvera(**kwargs)

    @tv.render()
    def _():
        print('_')
        tv.p()

if __name__=='__main__':
    fire.Fire(main)
