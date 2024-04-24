from pyaxidraw import axidraw


def check():
    ad = axidraw.AxiDraw()
    ad.plot_setup()
    ad.options.mode = 'cycle'
    ad.plot_run()


def plot(filename):
    ad = axidraw.AxiDraw()
    ad.plot_setup(filename)
    ad.plot_run()


if __name__ == '__main__':
    #check()
    plot('svgNew.svg')
