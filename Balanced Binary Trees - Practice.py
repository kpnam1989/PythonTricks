# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:22:28 2018

@author: nkieu
https://www.geeksforgeeks.org/convert-normal-bst-balanced-bst/
https://stackoverflow.com/questions/14001676/balancing-a-bst

There exists an efficient algorithm to re-balance a binary search tree
"""



class Node:
    def __init__(self, data):
        self.data = data
        self.name = str(self.data)
        self.left = None
        self.right = None
        self.parent = None
    
    def insert(self, newData):
        if self.data > newData:
            if self.left == None:
                tmp = Node(newData)
                self.left = tmp
                tmp.parent = self
            else:
                self.left.insert(newData)
        else:
            if self.right == None:
                tmp = Node(newData)
                self.right  = tmp
                tmp.parent = self
            else: 
                self.right.insert(newData)
    
    def InOrder(self):
        if self.left != None:
            self.left.InOrder()
            
        print(self.name)
        
        if self.right != None:
            self.right.InOrder()
    
    def PostOrder(self):
        tmp = []
        if self.left != None:
            tmp = tmp + self.left.PostOrder()
        if self.right != None:
            tmp = tmp + self.right.PostOrder()
        
        tmp.append(self.data)
        return tmp

    
    def InOrder_Iterative(self):
        res, toVisit = [], []
        current = self
        
        while current or len(toVisit) > 0:
            if current:
                toVisit.append(current)
                current = current.left
            else:
                tmp = toVisit.pop()
                res.append(tmp.name)
                
                if tmp.right:
                    current = tmp.right
                
        return res
    
    def PreOrder_Iterative(self):
        res, toVisit = [], [self]
        current = self
        
        while len(toVisit) > 0:
            current = toVisit.pop()
            res.append(current.name)
            
            if current.right:
                toVisit.append(current.right)
            if current.left:
                toVisit.append(current.left)
                
        return res
            
    def PostOrder_Iterative_Wiki(self):
        result, toVisit = [], []
        current, prevNode = self, None
        peekNode = None
        
        while len(toVisit) > 0 or current:
            if current:
                toVisit.append(current)
                current = current.left
            else:
                peekNode = toVisit[-1]
                if peekNode.right and peekNode.right != prevNode:
                    # move right
                    current = peekNode.right
                    prevNode = current
                else:
                    # move up
                    result.append(peekNode.name)
                    toVisit.pop()
                
                prevNode = peekNode
                    
        return result
            
    def __str__(self):
        return("Node " + self.name)

    def PreOrder_O1(self):
        # need parent pointer
        pass
    
    def BalancedTree(self):
        pass

def LCA(root, node1, node2):
    '''
    Least common ancestor
    '''
    from collections import namedtuple
    LCA = namedtuple("helper",["num","ancestor"])
    
    def LCA_helper(root, node1, node2):
        # This is 1 base case
        if not root:
            return LCA(0, None)
        
        leftResult = LCA_helper(root.left, node1, node2)
        if leftResult.num == 2:
            return leftResult
        
        rightResult = LCA_helper(root.right, node1, node2)
        if rightResult.num == 2:
            return rightResult
            
        # This is the 2nd case
        tmp = (root.data == node1.data or root.data == node2.data) + leftResult.num + rightResult.num
        
        if tmp == 2:
            return LCA(tmp, root)
        else:
            return LCA(tmp, None)
        
    return LCA_helper(root, node1, node2).ancestor
    
def RunningSum(root, weight):
    if not root.left and not root.right:
        if weight == root.data:
            return (True, [root.data])
    else:
        if root.left:
            leftRes = RunningSum(root.left, weight - root.data)
            if leftRes[0] == True:
                return True, [root.data] + leftRes[1]
        
        if root.right:
            rightRes = RunningSum(root.right, weight - root.data)
            if rightRes[0] == True:
                return True, [root.data] + rightRes[1]
    return (False, [])
        
    # search right
def ConstructTree_PreOrder():
    
    
    pass

def nextNode_InOrder(root):        
    tmp = root
    if tmp.right:
        tmp = tmp.right
        while tmp.left:
            tmp = tmp.left
        return tmp
    else:
        while tmp.parent and tmp.parent.left != tmp:
            tmp = tmp.parent
        return tmp.parent
    
        

root = Node(5)
root.insert(3)
root.insert(6)
root.insert(9)
root.insert(2)
root.insert(4)

print(root.InOrder_Iterative())
print(root.PreOrder_Iterative())
print(root.PostOrder_Iterative_Wiki())
print(root.PostOrder())
print(nextNode_InOrder(root.left).name)
print(nextNode_InOrder(root.left.left).name)
print(nextNode_InOrder(root.left.right).name)
print(nextNode_InOrder(root.right).name)

res = LCA(root, root.right, root.left.right).name
print("LCA", res)

print(RunningSum(root, 5 + 3 + 2)[1])
print(RunningSum(root, 5 + 6 + 9)[1])
print(RunningSum(root, 100)[1])