from PIL import Image

basedir = 'data/code/'

def combineImage(path, i, j, new_im):
	#opens an image
	im = Image.open(path)
	#convert mode so we can resize
	im = im.convert('RGB')
	#resize image
	im.thumbnail((200,200))
	#paste the image at location i,j
	new_im.paste(im, (i,j))
	return new_im

for index in xrange(1,21):
	filepath = basedir + `index` + '_'
	#creates a new empty image, RGB mode, and size 400 by 400.
	new_im = Image.new('RGB', (400,400))
	#Iterate through a 2 by 2 grid with 200 spacing, to place my image
	for i in xrange(0,400,200):
	    for j in xrange(0,400,200):

	    	if i == 0 and j == 0:
	    		path = filepath + 'green.tif'
	    	if i == 0 and j == 200:
	    		path = filepath + 'blue.tif'
	    	if i == 200 and j == 0:
	    		path = filepath + 'red.tif'
	    	if i == 200 and j == 200:
	    		path = filepath + 'yellow.tif'
	    	new_im = combineImage(path, i, j, new_im)
	new_im_path = basedir + "gen/" + `index` + ".jpg"
	new_im.save(new_im_path, "JPEG")
	print("Saved image: ", new_im_path)


