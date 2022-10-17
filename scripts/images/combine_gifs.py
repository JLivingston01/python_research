

import imageio
import numpy as np    
import os
os.listdir()
gifs = [ 'frame_animation_EMPTY.gif',
 'frame_animation_I_FORM.gif',
 'frame_animation_JUMBO.gif',
 'frame_animation_PISTOL.gif',
 'frame_animation_SHOTGUN.gif',
 'frame_animation_SINGLEBACK.gif',
 'frame_animation_WILDCAT.gif']


#Create reader object for the gif
gif1 = imageio.get_reader(gifs[0])
gif2 = imageio.get_reader(gifs[1])
gif3 = imageio.get_reader(gifs[2])
gif4 = imageio.get_reader(gifs[3])


#If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length()) 

#Create writer object
new_gif = imageio.get_writer('empty_i_jumbo_pistol.gif')

m = 35
n=125
for frame_number in range(number_of_frames):
    img1 = gif1.get_next_data()[m:576-m,n:576-n]
    img2 = gif2.get_next_data()[m:576-m,n:576-n]
    img3 = gif3.get_next_data()[m:576-m,n:576-n]
    img4 = gif4.get_next_data()[m:576-m,n:576-n]
    #here is the magic
    new_image = np.hstack((img1,img2,img3,img4))
    new_gif.append_data(new_image)

gif1.close()
gif2.close()  
gif3.close()  
gif4.close()  
new_gif.close()




#Create reader object for the gif
gif5 = imageio.get_reader(gifs[4])
gif6 = imageio.get_reader(gifs[5])
gif7 = imageio.get_reader(gifs[6])



#If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length()) 

#Create writer object
new_gif = imageio.get_writer('shot_single_wildcat.gif')

m = 35
n=125
for frame_number in range(number_of_frames):
    img5 = gif5.get_next_data()[m:576-m,n:576-n]
    img6 = gif6.get_next_data()[m:576-m,n:576-n]
    img7 = gif7.get_next_data()[m:576-m,n:576-n]
    #here is the magic
    new_image = np.hstack((img5,img6,img7))
    new_gif.append_data(new_image)

 
gif5.close()  
gif6.close()  
gif7.close()    
new_gif.close()





gifs = ['frame_animation_Cover 0 Man.gif',
 'frame_animation_Cover 1 Man.gif',
 'frame_animation_Cover 2 Man.gif',
 'frame_animation_Cover 2 Zone.gif',
 'frame_animation_Cover 3 Zone.gif',
 'frame_animation_Cover 4 Zone.gif',
 'frame_animation_Cover 6 Zone.gif',
 'frame_animation_Prevent Zone.gif']


#Create reader object for the gif
gif1 = imageio.get_reader(gifs[0])
gif2 = imageio.get_reader(gifs[1])
gif3 = imageio.get_reader(gifs[2])
gif4 = imageio.get_reader(gifs[3])


#If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length()) 

#Create writer object
new_gif = imageio.get_writer('cover_0M_1M_2M_2Z.gif')

m = 35
n=125
for frame_number in range(number_of_frames):
    img1 = gif1.get_next_data()[m:576-m,n:576-n]
    img2 = gif2.get_next_data()[m:576-m,n:576-n]
    img3 = gif3.get_next_data()[m:576-m,n:576-n]
    img4 = gif4.get_next_data()[m:576-m,n:576-n]
    #here is the magic
    new_image = np.hstack((img1,img2,img3,img4))
    new_gif.append_data(new_image)

gif1.close()
gif2.close()  
gif3.close()  
gif4.close()  
new_gif.close()




#Create reader object for the gif
gif5 = imageio.get_reader(gifs[4])
gif6 = imageio.get_reader(gifs[5])
gif7 = imageio.get_reader(gifs[6])
gif8 = imageio.get_reader(gifs[7])




#If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length()) 

#Create writer object
new_gif = imageio.get_writer('cover_3z_4z_6z_pz.gif')

m = 35
n=125
for frame_number in range(number_of_frames):
    img5 = gif5.get_next_data()[m:576-m,n:576-n]
    img6 = gif6.get_next_data()[m:576-m,n:576-n]
    img7 = gif7.get_next_data()[m:576-m,n:576-n]
    img8 = gif8.get_next_data()[m:576-m,n:576-n]
    #here is the magic
    new_image = np.hstack((img5,img6,img7,img8))
    new_gif.append_data(new_image)

 
gif5.close()  
gif6.close()  
gif7.close()   
gif8.close()    
new_gif.close()






