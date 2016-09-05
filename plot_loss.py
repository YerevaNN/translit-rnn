import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import codecs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log',default='log')
parser.add_argument('--window',default=100, type=int)
parser.add_argument('--ymax',default=0, type=float)
args = parser.parse_args()


def smoothen(values, window = 10):
	new_values = []
	i = 0
	while i + window <= len(values) :
		new_values.append(sum(values[i:i+window])/window)
		i += 1
	if i < len(values):
		new_values.append(sum(values[i:])/(len(values) - i ) )
	return new_values
def is_float(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

log = codecs.open(args.log).readlines()
losses = [float(line.split()[-7]) for line in log if len(line.split()) >= 7 and is_float(line.split()[-7]) ]
losses = smoothen(losses, args.window)
X = [float(line.split()[1]) for line in log if len(line.split()) >= 2 and is_float(line.split()[1]) ][-len(losses):]

if args.ymax > 0:
	plt.ylim(0, args.ymax)
	
plt.plot(X, losses)
plt.savefig(args.log + '.png')