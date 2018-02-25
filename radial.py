import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

import click

# AESTHETIC PARAMS
FONTSIZE = 8
ARCCOLOR = '.5'
TEXTOFFSET = .025
HIGHLIGHTCOLOR = '#ff0054'

# GEOMETRIC PARAMS
RADIUS = 1.
DIST = lambda P0, PN: np.linalg.norm(np.array(P0)-np.array(PN))
DISTANCECATEGORIES = [0, DIST([1, 0], 2 * [np.sqrt(2) / 2]), np.sqrt(2), DIST([1, 0],  [-np.sqrt(2) / 2, np.sqrt(2) / 2]), 2.0]
DISTANCECATEGORIES_BEZIERPARAMS = [1.2, 1.5, 1.8, 2.1]

# Bezier curve plotting
# adapted from https://plot.ly/python/chord-diagram/
def getbez(P0, PN):
	P0 = np.array(P0)
	PN = np.array(PN)

	def getdistancecat(d):
		k = 0
		while d > DISTANCECATEGORIES[k]: 
			k += 1
		return k - 1

	def deCasteljau(b, t): 
		N = len(b) 
		a = np.copy(b)
		for r in range(1, N): 
			a[:N-r,:] = (1 - t) * a[:N - r,:] + t * a[1:N - r + 1,:]                             
		return a[0,:]

	def BezierCurve(b, npoints = 100):
		t = np.linspace(0, 1, npoints)
		return np.array([deCasteljau(b, t[k]) for k in range(npoints)])

	d = DIST(P0, PN)
	K = getdistancecat(d)
	b = [P0, P0 / DISTANCECATEGORIES_BEZIERPARAMS[K], PN / DISTANCECATEGORIES_BEZIERPARAMS[K], PN]
	return BezierCurve(b)


@click.command()
@click.option('--file', '-f', default = 'network-longform.csv')
@click.option('--highlight', '-h', is_flag = True)
def draw(file, highlight):
	data = pd.read_csv(file) 
	nodes = pd.DataFrame(dict(node = pd.unique(data[['w1', 'w2']].values.ravel('K'))))

	# calculate circumference coords for nodes & labels
	n = nodes.shape[0]
	irange = np.arange(0, n)
	nodes['x'] = np.cos(2 * np.pi / n * irange) * RADIUS
	nodes['y'] = np.sin(2 * np.pi / n * irange) * RADIUS
	nodes['text_x'] = np.cos(2 * np.pi / n * irange) * (RADIUS + TEXTOFFSET)
	nodes['text_y'] = np.sin(2 * np.pi / n * irange) * (RADIUS + TEXTOFFSET)
	nodes = nodes.set_index("node")

	# set up the plot
	sns.set_style('white')
	fig, ax = plt.subplots()
	
	# record node coords in the network data
	data['xy_w1'] = data.w1.apply(lambda w1: (nodes.loc[w1].x, nodes.loc[w1].y))
	data['xy_w2'] = data.w2.apply(lambda w2: (nodes.loc[w2].x, nodes.loc[w2].y))

	def getangle(x, y):
		return np.degrees(np.arctan2(y, x))

	def gettextangle(row):
		return row.angle - 180 if row.x < 0 else row.angle

	nodes['angle'] = nodes.apply(lambda row: getangle(row.x, row.y), axis = 1)
	nodes['textangle'] = nodes.apply(lambda row: gettextangle(row), axis = 1)
	nodes = nodes.reset_index()

	if highlight:
		similarity_threshold = data.cosine_similarity.mean() + (2 * data.cosine_similarity.std())
		data['strong'] = data.cosine_similarity > similarity_threshold

	def plotarc(row):
		curve = getbez(row.xy_w1, row.xy_w2)
		curve_x, curve_y = zip(*curve)
		if highlight:
			ax.plot(curve_x, curve_y, alpha = row.cosine_similarity * .5 if not row.strong else row.cosine_similarity, lw = .65, color = ARCCOLOR if not row.strong else HIGHLIGHTCOLOR)
		else:
			ax.plot(curve_x, curve_y, alpha = row.cosine_similarity, lw = .65, color = ARCCOLOR)

	# plot neighbour curves
	data.apply(plotarc, axis = 1)

	# add text patches
	nodes.apply(lambda row: ax.text(row.text_x, row.text_y, row.node.decode('utf-8'), rotation=row.textangle, horizontalalignment = 'left' if row.x > 0 else 'right', verticalalignment = 'center', fontsize = 8, fontweight = 'light', rotation_mode="anchor"), axis = 1)

	ax.set(aspect=1)
	plt.xlim([-1.25, 1.25])
	plt.ylim([-1.25, 1.25])
	plt.xticks([])
	plt.yticks([])
	sns.despine(left = True, top=True, right=True, bottom=True)
	plt.savefig('radial-network.pdf', dpi = 400)

if __name__ == '__main__':
	draw()