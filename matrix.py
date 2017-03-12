import csv
from numpy import array

class Matrix:
     def __init__(self, filename):
          self.filename = filename
          self.mat = []        
          reader = csv.reader(file(filename))
          header = reader.next()
          self.colnames = header[1: ]
          self.rownames = []
          for line in reader:
               self.rownames.append(line[0])            
               self.mat.append(array(map(float, line[1: ])))
          self.mat = array(self.mat)

     def get_word_index(self, word):
          if word not in self.rownames:
               return -1
          return self.rownames.index(word)

     def get_vector_by_word(self, word):
          index = self.get_word_index(word)
          return self.mat[index, ]
        
     def get_value(self, i1, i2):
          return self.mat[i1, i2]

      
