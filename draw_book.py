import os
import numpy as np
import cv2
import json
import csv


# csv_folder = 'csv'
# list_file_csv = [os.path.join(csv_folder, file) for file in sorted(os.listdir(csv_folder))]
# for file in list_file_csv:
# 	name_file = file.split('/')[-1]
# 	name_file = name_file.split('.csv')[0]
# 	with open(file, 'r') as f:
# 		reader = csv.reader(f, delimiter=',')

# 		for row in reader:
# 			print(row)


# exit(0)






class re_polys(object):

	def __init__(self, listPoly):
		self.listPoly = listPoly
		self.sizes = []
		self.clusters = {}
		self.candidate_poly_error = []
		self.modified_poly = []

	def get_size_poly(self, poly):
		p1, p2, p3, p4 = poly
		w1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
		w2 = np.sqrt((p3[0] - p4[0])**2 + (p3[1] - p4[1])**2)

		h1 = np.sqrt((p4[0] - p1[0])**2 + (p4[1] - p1[1])**2)
		h2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)

		w = (w1+w2)/2
		h = (h1+ h2)/2

		return h, w

	def getListPolySize(self):

		for poly in self.listPoly:
			h, w = self.get_size_poly(poly)
			self.sizes.append(w)
		return self.sizes

	def cluster_polys(self):
		self.getListPolySize()
		n = len(self.sizes)
		i = 0
		sizes_clusters = self.sizes

		while n > 0:
			size_i = sizes_clusters[0]
			idxs_i = []
			for j, size in enumerate(sizes_clusters):
				if np.abs(size - size_i) / max(size, size_i) < 0.2:
					idxs_i.append(j)

			indexs = [idx for idx, size in enumerate(self.sizes) if size in sizes_clusters]
			idxs = np.array(indexs)[idxs_i]
			idxs = idxs.tolist()

			self.clusters[i] = idxs
			sizes_clusters = [sizes_clusters[j] for j in range(n) if not j in idxs_i]
			n = len(sizes_clusters)
			i += 1


		return self.clusters

	def get_errorPoly(self):
		self.cluster_polys()
		for cluster in self.clusters:
			print(self.clusters[cluster])


		len_clusters = [(i,len(self.clusters[i])) for i in self.clusters]
		len_clusters = sorted(len_clusters, key=lambda x:x[1])
		min_cluster = len_clusters[0][0]
		second_cluster = len_clusters[1][0]
		idxs_0 = self.clusters[min_cluster]
		idxs_1 = self.clusters[second_cluster]
		if len(idxs_0) < 3:
			self.candidate_poly_error.extend(idxs_0)
		else:
			for i, idx in enumerate(idxs_0):
				if i ==0:
					if not( idxs_0[i+1] - idx == 1 and idxs_0[i+2] - idx == 2):
						self.candidate_poly_error.append(idx)
				elif i != 0 and i!= len(idxs_0) - 1:
					if not(idx - idxs_0[i-1] == 1 and idxs_0[i+1] - idx==1):
						self.candidate_poly_error.append(idx)
				else:
					if not (idx - idxs_0[i-1] == 1 and idx - idxs_0[i-2] == 2):
						self.candidate_poly_error.append(idx)

		if len(idxs_1) < 3:
			self.candidate_poly_error.extend(idxs_1)
		else:
			for i, idx in enumerate(idxs_1):
				if i==0:
					if not(idxs_1[i+1] - idx ==1 and idxs_1[i+2] - idx ==2):
						self.candidate_poly_error.append(idx)
				elif i!= 0 and i!= len(idxs_1)-1:
					if not (idx - idxs_1[i-1] == 1 and idxs_1[i+1] - idx == 1):
						self.candidate_poly_error.append(idx)
				else:
					if not (idx - idxs_1[i-1] == 1 and idx - idxs_1[i-2] ==2):
						self.candidate_poly_error.append(idx)

		return self.candidate_poly_error

	def check_idx_cluster(self,idx):
		list_cluster = []
		for cluster in self.clusters:
			idxs = self.clusters[cluster]
			list_cluster.append(idxs)
		for i, idxs in enumerate(list_cluster):
			if idx < max(idxs) and idx > min(idxs):
				return i,idxs


	def modify_PolyError(self):
		self.get_errorPoly()

		merge_idx_error = []

		for idx in self.candidate_poly_error:
			if idx - 1 in self.candidate_poly_error or idx + 1 in self.candidate_poly_error:
				merge_idx_error.append(idx)
		merge_idx_error = np.array(merge_idx_error)
		merge_idxs_error = np.reshape(merge_idx_error, (int(merge_idx_error.shape[0]/2),2)).tolist()

		for idxs in merge_idxs_error:
			poly1, poly2 = np.array(self.listPoly)[idxs]

			p1 = poly1[0]
			p2 = poly2[1]
			p3 = poly2[2]
			p4 = poly1[3]
			poly = np.array([p1,p2,p3,p4])
			self.modified_poly.append(poly)

		not_merge_idxs = [idx for idx in self.candidate_poly_error if not idx in merge_idx_error]
		for idx in not_merge_idxs:
			size_poly = self.sizes[idx]
			poly = self.listPoly[idx]
			i, idxs = self.check_idx_cluster(idx)
			size_cluster = sum(np.array(self.sizes)[idxs])/ len(idxs)
			p1, p2,p3,p4 = poly
			if size_poly > 1.5*size_cluster:
				p2 = (p1 + p2) /2
				p4 = (p3+p4)/2
				self.modified_poly.append(np.array([p1, p2,p3,p4]))
			else:
				p2 = 2*p2 - p1
				p4 = 2*p4 - p3
				self.modified_poly.append(np.array([p1,p2,p3,p4]))

		return self.modified_poly


if __name__ == '__main__':
	# base_folder = 'bookPhoto/bboxs'
	image_folder = 'bookPhoto/input'
	csv_folder = 'csv'
	save_folder = 'modify_detectBook'
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)
	list_file_csv = sorted(os.listdir(csv_folder))
	for file in list_file_csv:
		path_csv = os.path.join(csv_folder, file)
		name_file = file.split('.csv')[0]
		name_img = name_file + '.jpeg'
		img_path = os.path.join(image_folder, name_img)
		img = cv2.imread(img_path)
		img1 = img.copy()


		polys = []

		with open(path_csv, 'r') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				row = row[:8]
				x1,y1,x2,y2,x3,y3,x4,y4 = row
				p1 = [int(x1), int(y1)]
				p2 = [int(x2), int(y2)]
				p3 = [int(x3), int(y3)]
				p4 = [int(x4), int(y4)]
				polys.append([p1,p2,p3,p4])
		polys = np.array(polys)
		error_list = re_polys(polys).get_errorPoly()

		for i,poly in enumerate(polys):
			poly = poly.reshape((-1,1,2))
			img = cv2.polylines(img, [poly], True, (0,255,0), 5)
			if not i in error_list:
				img1 = cv2.polylines(img1, [poly], True, (0,255,0),5)

		for i in error_list:
			poly = polys[i]
			poly = poly.reshape((-1,1,2))
			img1 = cv2.polylines(img1, [poly], True, (0,0,255), 5)
		cv2.imwrite(os.path.join(save_folder, name_img), img1)

		# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
		# cv2.namedWindow('img1', cv2.WINDOW_NORMAL)

		# cv2.imshow('img', img)
		# cv2.imshow('img1', img1)
		# cv2.waitKey()



	exit(0)
	list_file = sorted(os.listdir(base_folder))
	gt_files = [os.path.join(base_folder, file) for file in list_file]

	for file in gt_files:

		with open(file, 'r') as f:
			data = json.load(f)
			name_gt = file.split('/')[-1]
			name_img = name_gt.split('.json')[0] + '.jpeg' 
			img = cv2.imread(os.path.join(image_folder, name_img))
			img_copy = img.copy()
			img_copy2 = img.copy()

			shapes = data['shapes']
			polys = []

			for shape in shapes:
				poly = shape['points']
				poly = np.array(poly, np.int32)

				polys.append(poly)
				poly = poly.reshape((-1,1,2))

				img = cv2.polylines(img, [poly], True, (0,255,0),3)
				img_copy = cv2.polylines(img_copy, [poly], True, (0,255,0),3)

			# sizes = re_polys(polys).getListPolySize()
			error_list = re_polys(polys).get_errorPoly()
			for idx in error_list:
				poly = polys[idx]
				poly = poly.reshape((-1,1,2))
				img_copy = cv2.polylines(img_copy, [poly],True, (0,0,255), 5)

			# for i, poly in enumerate(polys):
			# 	if not i in error_list:
			# 		poly = poly.reshape((-1,1,2))
			# 		img_copy2 = cv2.polylines(img_copy2, [poly], True, (0,255,0),5)

			# modify_polys = re_polys(polys).modify_PolyError()
			# for poly in modify_polys:
			# 	poly = np.array(poly, np.int32)
			# 	poly = poly.reshape((-1,1,2))
			# 	img_copy2 = cv2.polylines(img_copy2, [poly], True, (255, 0,0),5)

			cv2.namedWindow('img', cv2.WINDOW_NORMAL)
			cv2.namedWindow('img_copy', cv2.WINDOW_NORMAL)
			cv2.namedWindow('img_copy2', cv2.WINDOW_NORMAL)

			cv2.imshow('img', img)
			cv2.imshow('img_copy', img_copy)
			cv2.imshow('img_copy2', img_copy2)

		cv2.waitKey()

