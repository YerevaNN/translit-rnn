import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import codecs
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--log',default='log')
parser.add_argument('--window',default=100, type=int)
parser.add_argument('--ymax',default=0, type=float)
args = parser.parse_args()


def smoothen(values, window = 10):
	new_values = []
	i = 0
	s = 0.0
	for i in range(len(values)):
		s += values[i]
		if i >= window:
			s -= values[i-window]
			new_values.append(s/window)
		else:
			new_values.append(s/(i+1))
	return new_values
	
def is_float(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

log = codecs.open(args.log).readlines()
losses = [float(line.split()[5][:-1]) for line in log if len(line.split()) > 5 and is_float(line.split()[5][:-1]) ]
val_losses = [float(line.split()[-1]) for line in log if len(line.split()) > 0 and line.split()[0] == 'Validation' ]

losses = smoothen(losses, args.window)
val_losses = smoothen(val_losses, max(args.window * len(val_losses)/len(losses), 1) )

x = np.linspace(0, 1, len(losses)) 
val_x = np.linspace(0, 1, len(val_losses))


if args.ymax > 0:
	plt.ylim(0, args.ymax)

plt.plot(val_x, val_losses)
plt.plot(x, losses)
plt.savefig(args.log + '.png')
