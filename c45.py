import math


class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, path_to_data, path_to_names):
		self.file_path_to_data = path_to_data
		self.filePathToNames = path_to_names
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None

	def fetch_data(self):
		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			# add attributes
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				self.attrValues[attribute] = values
		self.numAttributes = len(self.attrValues.keys())
		self.attributes = list(self.attrValues.keys())
		with open(self.file_path_to_data, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				if row != [] or row != [""]:
					self.data.append(row)

	def preprocess_data(self):
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if not self.is_attr_discrete(self.attributes[attr_index]):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def print_tree(self):
		self.print_node(self.tree)

	def print_node(self, node, indent=""):
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent + node.label + " = " + self.attributes[index] + " : " + child.label)
					else:
						print(indent + node.label + " = " + self.attributes[index] + " : ")
						self.print_node(child, indent + "	")
			else:
				#numerical
				left_child = node.children[0]
				right_child = node.children[1]
				if left_child.isLeaf:
					print(indent + node.label + " <= " + str(node.threshold) + " : " + left_child.label)
				else:
					print(indent + node.label + " <= " + str(node.threshold)+" : ")
					self.print_node(left_child, indent + "	")

				if right_child.isLeaf:
					print(indent + node.label + " > " + str(node.threshold) + " : " + right_child.label)
				else:
					print(indent + node.label + " > " + str(node.threshold) + " : ")
					self.print_node(right_child, indent + "	")


	def generate_tree(self):
		self.tree = self.recursive_generate_tree(self.data, self.attributes)

	def recursive_generate_tree(self, curData, curAttributes):
		all_same = self.all_same_class(curData)

		if len(curData) == 0:
			# Fail
			return Node(True, "Fail", None)
		elif all_same is not False:
			# return a node with that class
			return Node(True, all_same, None)
		elif len(curAttributes) == 0:
			# return a node with the majority class
			majClass = self.get_maj_class(curData)
			return Node(True, majClass, None)
		else:
			(best,best_threshold,splitted) = self.split_attribute(curData, curAttributes)
			remaining_attributes = curAttributes[:]
			remaining_attributes.remove(best)
			node = Node(False, best, best_threshold)
			node.children = [self.recursive_generate_tree(subset, remaining_attributes) for subset in splitted]
			return node

	def get_maj_class(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]

	def all_same_class(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def is_attr_discrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
			return False
		else:
			return True

	def gain(self, unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	def entropy(self, data_set):
		S = len(data_set)
		if S == 0:
			return 0
		num_classes = [0 for i in self.classes]
		for row in data_set:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
		num_classes = [x/S for x in num_classes]
		ent = 0
		for num in num_classes:
			ent += num*self.log(num)
		return ent*-1

	def split_attribute(self, cur_data, cur_attributes):
		splitted = []
		max_ent = -1 * float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in cur_attributes:
			index_of_attribute = self.attributes.index(attribute)
			if self.is_attr_discrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				values_for_attribute = self.attrValues[attribute]
				subsets = [[] for a in values_for_attribute]
				for row in cur_data:
					for index in range(len(values_for_attribute)):
						if row[index] == values_for_attribute[index]:
							subsets[index].append(row)
							break
				e = self.gain(cur_data, subsets)
				if e > max_ent:
					max_ent = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				cur_data.sort(key = lambda x: x[index_of_attribute])
				for j in range(0, len(cur_data) - 1):
					if cur_data[j][index_of_attribute] != cur_data[j + 1][index_of_attribute]:
						threshold = (cur_data[j][index_of_attribute] + cur_data[j + 1][index_of_attribute]) / 2
						less = []
						greater = []
						for row in cur_data:
							if(row[index_of_attribute] > threshold):
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(cur_data, [less, greater])
						if e >= max_ent:
							splitted = [less, greater]
							max_ent = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x,2)


class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []


