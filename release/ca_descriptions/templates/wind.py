import numpy as np

DIRECTIONS = {"NW", "N", "NE", "W", "E", "SW", "S", "SE"}

def opposite(direction): #gets opposite direction e.g opposite of NE is SW
	list_directions = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
	item_index = np.where(list_directions==direction)[0]+4
	return list_directions[item_index % len(list_directions)]



def wind_speed(direction, speed):
	if direction in DIRECTIONS:
		list_directions = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
		item_index = np.where(list_directions==direction)[0]
		listWeights = np.zeros(8)
		weight_interval = speed/100
		weight = weight_interval*2 #initialises weight
		wrapped = False
		for x in range(8):       #goes through array, including wrapping round and weights the directions
			listWeights[(x+item_index) % len(list_directions)] = 1+weight
			if weight > -2*weight_interval and not wrapped:		
				weight= weight-weight_interval 
			else:
				wrapped = True
				weight = weight+weight_interval

		rearranged_index = [7,0,1,6,2,5,4,3] #rearranges list so is in same order as the CA programme

		return listWeights[rearranged_index]
def k_wind(speed, angle):
	return np.exp(0.1783*speed* np.cos(np.deg2rad(angle)))

def wind_speed_rvalue(direction, speed):
	if direction in DIRECTIONS:
		list_directions = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
		item_index = np.where(list_directions==direction)[0]
		listWeights = np.zeros(8)
		angle_interval = 45
		angle = 0 #initialises weight
		wrapped = False
		for x in range(8):       #goes through array, including wrapping round and weights the directions
			listWeights[(x+item_index) % len(list_directions)] = k_wind(speed, angle)
			angle = angle + angle_interval
			# if angle > -2*angle_interval and not wrapped:		
			# 	angle = angle-angle_interval 
			# else:
			# 	wrapped = True
			# 	weight = weight+weight_interval

		rearranged_index = [7,0,1,6,2,5,4,3] #rearranges list so is in same order as the CA programme

		return listWeights[rearranged_index]

			




	# 	for x in listDirections:
	# 		if x==direction:                      #if x is the same as the direction highest weighting
	# 			listWeights.append(1+speed/100*2)
	# 		elif x == direction[0] || x == direction[1]:   #if direction is a diagonal then weight the ones beside it  
	# 			listWeights.append(1+speed/100)                 # e.g. NE, weight N & E a bit
	# 		elif x[0] == direction:				           #if direction is on the perpendicualr, weight adj
	# 			listWeights.append(1+speed/100)                  # e.g. N weight NW & NE a bit
	# 		elsif x == opposite_dir:			           #if x is the opposite direction lowest weighting 
	# 			listWeights.append(1-speed/100*2) 
	# 		elif x == opposite_dir[0] || x == opposite_dir[1]:  #if opp_dir is a diagonal then weight the ones beside it low  
	# 			listWeights.append(1-speed/100) 				  # e.g. NE, weight N & E a bit low			
	# 		elif x[0] == opposite_dir:				           #if opp_dir is on the perpendicualr, weight adj lol
	# 			listWeights.append(1-speed/100)         

	# 		else:
	# 			listWeights.append(1)
	# 	return listWeights
	# else:
	# 	return "no"

print(wind_speed("S", 20))
print(wind_speed_rvalue("S", 6))

