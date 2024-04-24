import svgutils

input = 'Sound_100.svg'
scale = 0.45

svg = svgutils.transform.fromfile(input)
originalSVG = svgutils.compose.SVG(input)
originalSVG.scale(scale)
figure = svgutils.compose.Figure(float(svg.width) * scale, float(svg.height) * scale, originalSVG)
figure.save('svgNew.svg')
