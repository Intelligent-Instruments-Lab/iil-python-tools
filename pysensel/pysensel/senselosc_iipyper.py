import taichi as ti
from iipyper import OSC, run, repeat, cleanup

class SenselOSC():
    """
    https://github.com/tai-studio/senselosc/readme.md
    """
    def __init__(self):
        self._dims = {
            'x': 230.0,
            'y': 125.0,
            'f': 4096.0,
            'a': 4096.0
        }
        self._max_contacts = 16
        self.contacts = [self._dims] * self._max_contacts

    def handle_contact(self, args):
        """
        Basic values for each registered contact.

        /contact index id state x y force area dist wdist 
                orientation  major_axis  minor_axis
        """
        index = args[0] # [int]   device index (currently always 0)
        id    = args[1] # [int]   contact id (0..15)
        self.contacts[id] = {
            'state': args[2],  # [int]   one of invalid(0), start(1), move(2), end(3) 
            'x': args[3]/self._dims['x'],  # [float] x-coordinate in [mm]
            'y': args[4]/self._dims['y'],  # [float] y-coordinate in [mm]
            'f': args[5]/self._dims['f'],  # [float] sum of pressure applied [g] 
            'a': args[6]/self._dims['a'],  # [int]   covered area [sensels]
            'dist':        args[7],  # [float] distance to average position [mm]
            'wdist':       args[8],  # [float] distance to weighted average position [mm]
            'orientation': args[9],  # [float] orientation of bounding elipsis [deg] (0..360)
            'major_axis':  args[10], # [float] major axis length of bounding elipsis [mm]
            'minor_axis':  args[11]  # [float] minor axis length of bounding elipsis [mm]
        }

    # def __handle_contact_avg(*args):
    #     """
    #     Average values for all currently registered contacts. 
    #     If messages are sent in bundles, this message precedes 
    #     the messages for the individual contacts. If no contacts 
    #     are detected, a message with num_contacts = 0 is sent once.

    #     /contactAvg index num_contacts x y avg_force avg_dist area 
    #                 x_w y_w total_force avg_wdist

    #     index           [int]   device index (currently always 0)
    #     num_contacts    [int]   number of contacts
    #     x               [float] average x-coordinate over all contacts in [mm]
    #     y               [float] average y-coordinate over all contacts in [mm]
    #     avg_force       [float] average pressure applied [g] 
    #     avg_dist        [float] distance to average position [mm]
    #     area            [int]   covered area [sensels]
    #     w_x             [float] force-weigthed average x-coordinate over all contacts in [mm]
    #     w_y             [float] force-weigthed average y-coordinate over all contacts in [mm]
    #     total_force     [float] sum of pressure applied [g] 
    #     avg_wdist       [float] distance to force-weighted average position [mm]
    #     """
    #     pass

    # def __handle_contact_delta(*args):
    #     """
    #     Delta-values for each registered contact.

    #     /contactDelta index id state delta_x delta_y 
    #                 delta_force delta_area
        
    #     index           [int]   device index (currently always 0)
    #     id              [int]   contact id (0..15)
    #     state           [int]   one of invalid(0), start(1), move(2), end(3) 
    #     num_contacts    [int]   number of contacts
    #     delta_x         [float] x displacement [mm]
    #     delta_y         [float] y displacement [mm]
    #     delta_force     [float] change of force [g]
    #     delta_area      [int]   change of covered area [sensels]
    #     """
    #     pass

    # def __handle_contact_bb(*args):
    #     """
    #     A bounding box for each registered contact.

    #     /contactBB index id state min_x min_y max_x max_y

    #     index           [int]   device index (currently always 0)
    #     id              [int]   contact id (0..15)
    #     state           [int]   one of invalid(0), start(1), move(2), end(3) 
    #     min_x           [float] upper-left x-coordinate of bounding-box [mm] 
    #     min_y           [float] upper-left y-coordinate of bounding-box [mm] 
    #     max_x           [float] lower-right x-coordinate of bounding-box [mm] 
    #     max_y           [float] lower-right y-coordinate of bounding-box [mm] 
    #     """
    #     pass

    # def __handle_contact_peak(*args):
    #     """
    #     Peak values for each registered contact.

    #     /contactPeak index id state peak_x peak_y peak_force

    #     index           [int]   device index (currently always 0)
    #     id              [int]   contact id (0..15)
    #     state           [int]   one of invalid(0), start(1), move(2), end(3) 
    #     peak_x          [float] x-coordinate of pressure peak [mm]
    #     peak_y          [float] y-coordinate of pressure peak [mm]
    #     peak_force      [float] force at pressure peak [g]
    #     """
    #     pass

    # def __handle_sync(*args):
    #     """
    #     Each processed frame of updated information is concluded 
    #     by a sync message that can be used to update functionality 
    #     using all sent values.

    #     /sync index updated_0 ... updated_15

    #     index           [int]   device index (currently always 0)
    #     updated_X       [int]   1 if contact id was updated, 0 otherwise
    #     """
    #     pass

def main(host="127.0.0.1", port=7006):
    osc = OSC(host, port)
    sensel = SenselOSC()
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    # n = 4096
    # # obstacles = Obstacles(x, y)
    # boids = Boids(x, y, n, colormode='rgb', species=10)

    @osc.args("/contact")
    def _(address, *args):
        sensel.handle_contact(args)
        print(sensel.contacts)

    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()
    while window.running:
        # update(boids) # jurigged: costs 10fps
        canvas.set_image()
        window.show()

if __name__ == '__main__':
    run(main)
