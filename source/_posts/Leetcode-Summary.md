---
title: Leetcode_Summary
date: 2020-01-20 13:02:27
tags: Programming
---

Sources:
1. summer
2. 1-50
3. Top Interview Questions (before 200)

## DFS + memo

### 98. Validate Binary Search Tree (Medium) [@](https://leetcode.com/problems/validate-binary-search-tree/)

> Given a binary tree, determine if it is a valid binary search tree (BST).
>
> Assume a BST is defined as follows:
>
> - The left subtree of a node contains only nodes with keys **less than** the node's key.
> - The right subtree of a node contains only nodes with keys **greater than** the node's key.
> - Both the left and right subtrees must also be binary search trees.
>

**Solution 1 Recursion**

```java

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private boolean helper(TreeNode node, Integer lower, Integer upper) {
        if (node == null) return true;
        
        int val = node.val;
        if (lower != null && val <= lower) return false;
        if (upper != null && val >= upper) return false;
        
        if (!helper(node.left, lower, val)) return false;
        if (!helper(node.right, val, upper)) return false;
        return true;
    }
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }
}
```
**Solution 2 Iteration**

```java
    LinkedList<TreeNode> stack = new LinkedList();
    LinkedList<Integer> uppers = new LinkedList(), lowers = new LinkedList();
    
    private void update(TreeNode root, Integer lower, Integer upper) {
        stack.add(root);
        uppers.add(upper);
        lowers.add(lower);
    }
    public boolean isValidBST(TreeNode root) {
        Integer lower = null, upper = null, val;
        update(root, lower, upper);
        
        while (!stack.isEmpty()) {
            root = stack.poll();
            lower = lowers.poll();
            upper = uppers.poll();
            
            if (root == null) continue;
            val = root.val;
            if (lower != null && val <= lower) return false;
            if (upper != null && val >= upper) return false;
            update(root.left, lower, val);
            update(root.right, val, upper);
        }
        return true;
    }
```
**Solution 3 Inorder Traversal**

```java
    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> stack = new Stack();
        double inorder = - Double.MAX_VALUE;
        
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.val <= inorder) return false;
            inorder = root.val;
            root = root.right;
        }
        return true;
    }
```

### 101. Symmetric Tree (Easy) [@](https://leetcode.com/problems/symmetric-tree/)

> Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
>
> For example, this binary tree `[1,2,2,3,4,4,3]` is symmetric:
>
> ```
>  1
> / \
> 2   2
> / \ / \
> 3  4 4  3
> ```
>
> 
>
> But the following `[1,2,2,null,3,null,3]` is not:
>
> ```
>  1
> / \
> 2   2
> \   \
> 3    3
> ```

**Solution 1 Recursion**

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val) && isMirror(t1.right, t2.left) && isMirror(t1.left, t2.right);
    }
    public boolean isSymmetric(TreeNode root) {
        if (root == null) 
            return true;
        else
            return isMirror(root.left, root.right);
    }
}
```
**Solution 2 Iteration**

```java
public boolean isSymmetric(TreeNode root) {
    if (root == null) return true;

    Queue<TreeNode> q = new LinkedList();
    q.add(root.left);
    q.add(root.right);
    while (!q.isEmpty()) {
        TreeNode t1 = q.poll();
        TreeNode t2 = q.poll();
        if (t1 == null && t2 == null) continue;
        if (t1 == null || t2 == null) return false;
        if (t1.val != t2.val) return false;
        q.add(t1.left);
        q.add(t2.right);
        q.add(t1.right);
        q.add(t2.left);
    }
    return true;
}
```

### 104. Maximum Depth of Binary Tree (Easy) [@](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

> Given a binary tree, find its maximum depth.
>
> The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
>
> **Note:** A leaf is a node with no children.
>
> **Example:**
>
> Given binary tree `[3,9,20,null,null,15,7]`,
>
> ```
>  3
> / \
> 9  20
>  /  \
> 15   7
> ```
>
> return its depth = 3.

**Solution 1 Recursion**

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```
**Solution 2 Iteration (BFS)**

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        
        LinkedList<TreeNode> queue = new LinkedList<TreeNode>();
        int level = 0;
        queue.add(root);
        int curNum = 1, nextNum = 0;
        
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            curNum--;
            if (node.left != null) {
                nextNum++;
                queue.add(node.left);
            }
            if (node.right != null) {
                nextNum++;
                queue.add(node.right);
            }
            if (curNum == 0) {
                curNum = nextNum;
                nextNum = 0;
                level++;
            }
        }
        return level;
    }
}
```
### 105.  Construct Binary Tree from Preorder and Inorder Traversal (Medium) [@](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

> Given preorder and inorder traversal of a tree, construct the binary tree.
>
> **Note:**
> You may assume that duplicates do not exist in the tree.
>
> For example, given
>
> ```
> preorder = [3,9,20,15,7]
> inorder = [9,3,15,20,7]
> ```
>
> Return the following binary tree:
>
> ```
>  3
> / \
> 9  20
>  /  \
> 15   7
> ```

**Solution Divide and Conquer + Recursion** 

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        TreeNode root = createTree(preorder, 0, preorder.length-1, inorder, 0, inorder.length-1);
        return root;
    }
    private TreeNode createTree(int[] preorder, int startPre, int endPre, int[] inorder, int startIn, int endIn) {
        if (startPre > endPre || startIn > endIn) return null;
        TreeNode root = new TreeNode(preorder[startPre]);
        
        for (int i = startIn; i <= endIn; i++) {
            if (inorder[i] == preorder[startPre]) {
                //i-startIn是左子树长度
                root.left = createTree(preorder, startPre + 1, startPre + i - startIn, inorder, startIn, i-1);
                //右子树开始节点是从左子树开始节点加上左子树的长度
                root.right = createTree(preorder, startPre + 1 + i - startIn, endPre, inorder, i + 1, endIn);
            }
        }
        return root;
    }
}
```
### 108. Convert Sorted Array to Binary Search Tree (Easy) [@](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

> Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
>
> For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of *every* node never differ by more than 1.
>
> **Example:**
>
> ```
> Given the sorted array: [-10,-3,0,5,9],
> 
> One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:
> 
>    0
>   / \
> -3   9
> /   /
> -10  5
> ```

**Solution 1 Recursion**

```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return dfs(nums, 0, nums.length-1);
    }
    private TreeNode dfs(int[] nums, int start, int end) {
        if (start > end) return null;
        int mid = (start + end) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = dfs(nums, start, mid - 1);
        root.right = dfs(nums, mid + 1, end);
        return root;
    }
}
```
### 116. Populating Next Right Pointers in Each Node (Medium) [@](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

> You are given a **perfect binary tree** where all leaves are on the same level, and every parent has two children.
>
> Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to `NULL`.
>
> Initially, all next pointers are set to `NULL`.
>
> **Follow up:**
>
> - You may only use constant extra space.
> - Recursive approach is fine, you may assume implicit stack space does not count as extra space for this problem.

**Solution 1 Recursion**

这道题解法还是挺直白的，如果当前节点有左孩子，那么左孩子的next就指向右孩子。如果当前节点有右孩子，那么判断，如果当前节点的next是null，说明当前节点已经到了最右边，那么右孩子也是最右边的，所以右孩子指向null。如果当前节点的next不是null，那么当前节点的右孩子的next就需要指向当前节点next的左孩子。递归求解就好。

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/
class Solution {
    public Node connect(Node root) {
        if (root == null) return null;
        if (root.left != null) {
            root.left.next = root.right;
        }
        if (root.right != null) {
            if (root.next != null) {
                root.right.next = root.next.left;
            }else {
                root.right.next = null;
            }
        }
        connect(root.left);
        connect(root.right);
        return root;
    }
}
```

### 124. Binary Tree Maximum Path Sum (Hard) [@](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

> Given a **non-empty** binary tree, find the maximum path sum.
>
> For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain **at least one node** and does not need to go through the root.
>
> **Example 1:**
>
> ```
> Input: [1,2,3]
> 
>     1
>    / \
>   2   3
> 
> Output: 6
> ```
>
> **Example 2:**
>
> ```
> Input: [-10,9,20,null,null,15,7]
> 
> -10
> / \
> 9  20
>  /  \
> 15   7
> 
> Output: 42
> ```

- 递归的思想，DFS，从下到上
- 每个节点可以与其左右节点结合，但每个节点作为子节点返回时，只能选去该节点的值和其较大子节点的值的和返回
Solution
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    int res = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        helper(root);
        return res;
    }
    private int helper(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(helper(root.left), 0);
        int right = Math.max(helper(root.right), 0);
        res = Math.max(res, left + right + root.val);
        return Math.max(left, right) + root.val;
    }
}
```

### 130. Surrounded Regions (Medium) [@](https://leetcode.com/problems/surrounded-regions/)

> Given a 2D board containing `'X'` and `'O'` (**the letter O**), capture all regions surrounded by `'X'`.
>
> A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.
>
> **Example:**
>
> ```
> X X X X
> X O O X
> X X O X
> X O X X
> ```
>
> After running your function, the board should be:
>
> ```
> X X X X
> X X X X
> X X X X
> X O X X
> ```
>
> **Explanation:**
>
> Surrounded regions shouldn’t be on the border, which means that any `'O'` on the border of the board are not flipped to `'X'`. Any `'O'` that is not on the border and it is not connected to an `'O'` on the border will be flipped to `'X'`. Two cells are connected if they are adjacent cells connected horizontally or vertically.

**Solution Recursion**

```java
class Solution {
    public void solve(char[][] board) {
        if (board == null) return;
        int rows = board.length;
        if (rows <= 0) return;
        int cols = board[0].length;
        if (cols <= 0) return;
        // 找到边缘‘O’
        for (int i = 0; i < rows; i++) {
            if (board[i][0] == 'O')
                dfs(board, i, 0);
            if (board[i][cols-1] == 'O')
                dfs(board, i, cols-1);
        }
        for (int i = 0; i < cols; i++) {
            if (board[0][i] == 'O')
                dfs(board, 0, i);
            if (board[rows-1][i] == 'O')
                dfs(board, rows-1, i);
        }
        
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == '#')
                    board[i][j] = 'O';
                else if (board[i][j] == 'O')
                    board[i][j] = 'X';
            }
        }
    }
    //每遇到‘O’后，向四个方向递归搜索，所有相邻‘O’变为‘#’
    private void dfs(char[][] board, int i, int j) {
        if (board[i][j] == 'O') {
            board[i][j] = '#';
		// 跳过四周边缘
            if (i < board.length - 2)
                dfs(board, i + 1, j);
            if (i > 1)
                dfs(board, i - 1, j);
            if (j < board[0].length - 2)
                dfs(board, i, j + 1);
            if (j > 1) 
                dfs(board, i, j - 1);
        }
    }
}
```

### 200. Number of Islands (Medium) [@](https://leetcode.com/problems/number-of-islands/)

> Given a 2d grid map of `'1'`s (land) and `'0'`s (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
>
> **Example 1:**
>
> ```
> Input:
> 11110
> 11010
> 11000
> 00000
> 
> Output: 1
> ```
>
> **Example 2:**
>
> ```
> Input:
> 11000
> 11000
> 00100
> 00011
> 
> Output: 3
> ```

**Solution DFS + Recursion**

- 采用DFS，访问过的‘1’转为‘0’，继续遍历
```java
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int rows = grid.length;
        int cols = grid[0].length;
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1'){
                    count++;
                    dfs(grid, i, j);    
                }
            }
        }
        return count;
    }
    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i > grid.length-1 || j < 0 || j > grid[0].length-1)
            return;
        if (grid[i][j] == '0') {
            return;
        }else if (grid[i][j] == '1') {
            grid[i][j] = '0';
            dfs(grid, i-1, j);
            dfs(grid, i+1, j);
            dfs(grid, i, j-1);
            dfs(grid, i, j+1);
        }
    }
}
```

### 207. Course Schedule (Medium) [@](https://leetcode.com/problems/course-schedule/)

> There are a total of *n* courses you have to take, labeled from `0` to `n-1`.
>
> Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: `[0,1]`
>
> Given the total number of courses and a list of prerequisite **pairs**, is it possible for you to finish all courses?
>
> **Example 1:**
>
> ```
> Input: 2, [[1,0]] 
> Output: true
> Explanation: There are a total of 2 courses to take. 
>           To take course 1 you should have finished course 0. So it is possible.
> ```
>
> **Example 2:**
>
> ```
> Input: 2, [[1,0],[0,1]]
> Output: false
> Explanation: There are a total of 2 courses to take. 
>           To take course 1 you should have finished course 0, and to take course 0 you should
>           also have finished course 1. So it is impossible.
> ```
>
> **Note:**
>
> 1. The input prerequisites is a graph represented by **a list of edges**, not adjacency matrices. Read more about [how a graph is represented](https://www.khanacademy.org/computing/computer-science/algorithms/graph-representation/a/representing-graphs).
> 2. You may assume that there are no duplicate edges in the input prerequisites.

**Solution Topology**

- 此问题等价于图中是否有无环的存在（拓扑排序解决问题）
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] indegree = new int[numCourses]; 
        //初始化图，利用hashmap
        for (int i = 0; i < prerequisites.length; i++) {
            int s_node = prerequisites[i][0];
            int e_node = prerequisites[i][1];
            if (!map.containsKey(s_node))
                map.put(s_node, new ArrayList<>());
            map.get(s_node).add(e_node);
            indegree[e_node]++;//更新每个点的入度
        }
        //储存所有入度为0的节点->拓扑排序起始点
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0)
                q.offer(i);
        }
        //计算可拓扑排序的节点个数
        int count = 0;
        while (!q.isEmpty()) {
            int val = q.poll();
            count++;
            
            if (map.containsKey(val)) {
                List<Integer> tmp = map.get(val);
                for (int i = 0; i < tmp.size(); i++) {
                    int idx = tmp.get(i);
                    indegree[idx]--;
                    if (indegree[idx] == 0)
                        q.offer(idx);
                }
            }
        }
        return count == numCourses;
    }
}
```

### 210. Course Schedule II (Medium) [@](https://leetcode.com/problems/course-schedule-ii/)

> There are a total of *n* courses you have to take, labeled from `0` to `n-1`.
>
> Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: `[0,1]`
>
> Given the total number of courses and a list of prerequisite **pairs**, return the ordering of courses you should take to finish all courses.
>
> There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.
>
> **Example 1:**
>
> ```
> Input: 2, [[1,0]] 
> Output: [0,1]
> Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
>           course 0. So the correct course order is [0,1] .
> ```
>
> **Example 2:**
>
> ```
> Input: 4, [[1,0],[2,0],[3,1],[3,2]]
> Output: [0,1,2,3] or [0,2,1,3]
> Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
>           courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
>           So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .
> ```
>
> **Note:**
>
> 1. The input prerequisites is a graph represented by **a list of edges**, not adjacency matrices. Read more about [how a graph is represented](https://www.khanacademy.org/computing/computer-science/algorithms/graph-representation/a/representing-graphs).
> 2. You may assume that there are no duplicate edges in the input prerequisites.

**Solution Topology**

```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] indegree = new int[numCourses]; 
        int[] res = new int[numCourses];
        //初始化图，利用hashmap
        for (int i = 0; i < prerequisites.length; i++) {
            int s_node = prerequisites[i][0];
            int e_node = prerequisites[i][1];
            if (!map.containsKey(s_node))
                map.put(s_node, new ArrayList<>());
            map.get(s_node).add(e_node);
            indegree[e_node]++;//更新每个点的入度
        }
        //储存所有入度为0的节点->拓扑排序起始点
        int index = numCourses - 1;
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
                res[index--] = i;
            }
        }
        //拓扑排序
        while (!q.isEmpty()) {
            int val = q.poll();
            //获取val指向的节点
            if (map.containsKey(val)) {
                List<Integer> tmp = map.get(val);
                for (int i = 0; i < tmp.size(); i++) {
                    int idx = tmp.get(i);
                    indegree[idx]--;
                    if (indegree[idx] == 0) {
                        q.offer(idx);
                        res[index--] = idx;
                    }
                }
            }
        }
        if (index != -1)
            return new int[0];
        else
            return res;
    }
}
```
## BFS

### 102. Binary Tree Level Order Traversal (Medium) [@](https://leetcode.com/problems/binary-tree-level-order-traversal/)

> Given a binary tree, return the *level order* traversal of its nodes' values. (ie, from left to right, level by level).
>
> For example:
> Given binary tree `[3,9,20,null,null,15,7]`,
>
> ```
>  3
> / \
> 9  20
>  /  \
> 15   7
> ```
>
> 
>
> return its level order traversal as:
>
> ```
> [
> [3],
> [9,20],
> [15,7]
> ]
> ```

**Solution 1 Recursion**

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        helper(root, res, 0);
        return res;
    }
    private void helper(TreeNode root, List<List<Integer>> res, int level) {
        if (root == null) return;
        if (res.size() < level+1) {
            res.add(new ArrayList<Integer> ());
        }
        res.get(level).add(root.val);
        
        helper(root.left, res, level+1);
        helper(root.right, res, level+1);
    }
}
```
**Solution 2 Iteration (Queue)**

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (root == null) return res;
        
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        int level = 0;
        while (!queue.isEmpty()) {
            //start current level
            res.add(new ArrayList<Integer>());
            //num of elements in current level
            int len = queue.size();
            
            for (int i = 0; i < len; i++) {
                TreeNode node = queue.poll();
                //get the val in each level
                res.get(level).add(node.val);
                //add child nodes to queue
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            //go to next level
            level++;
        }
        return res;
    }
}
```

### 103. Binary Tree Zigzag Level Order Traversal (Medium) [@](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

> Given a binary tree, return the *zigzag level order* traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
>
> For example:
> Given binary tree `[3,9,20,null,null,15,7]`,
>
> ```
>  3
> / \
> 9  20
>  /  \
> 15   7
> ```
>
> 
>
> return its zigzag level order traversal as:
>
> ```
> [
> [3],
> [20,9],
> [15,7]
> ]
> ```

**Solution Recursion**

- based on 102, add a flag to identify reverse
```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        helper(root, res, 0, false);
        return res;
    }
    private void helper(TreeNode root, List<List<Integer>> res, int level, boolean flag) {
        if (root == null) return;
        if (res.size() < level+1) {
            res.add(new LinkedList<Integer> ());
        }
        if (flag) {
            //convert to LinkedList
            ((LinkedList<Integer>)res.get(level)).addFirst(root.val);
        }else {
            res.get(level).add(root.val);
        }
        
        helper(root.left, res, level+1, !flag);
        helper(root.right, res, level+1, !flag);
    }
}
```

### 127. Word Ladder (Medium) [@](https://leetcode.com/problems/word-ladder/)

> Given two words (*beginWord* and *endWord*), and a dictionary's word list, find the length of shortest transformation sequence from *beginWord* to *endWord*, such that:
>
> 1. Only one letter can be changed at a time.
> 2. Each transformed word must exist in the word list. Note that *beginWord* is *not* a transformed word.
>
> **Note:**
>
> - Return 0 if there is no such transformation sequence.
> - All words have the same length.
> - All words contain only lowercase alphabetic characters.
> - You may assume no duplicates in the word list.
> - You may assume *beginWord* and *endWord* are non-empty and are not the same.
>
> **Example 1:**
>
> ```
> Input:
> beginWord = "hit",
> endWord = "cog",
> wordList = ["hot","dot","dog","lot","log","cog"]
> 
> Output: 5
> 
> Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
> return its length 5.
> ```
>
> **Example 2:**
>
> ```
> Input:
> beginWord = "hit"
> endWord = "cog"
> wordList = ["hot","dot","dog","lot","log"]
> 
> Output: 0
> 
> Explanation: The endWord "cog" is not in wordList, therefore no 
> ```

**Solution 1 BFS (Time Limit Exceeded)**

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return 0;
        Queue<String> queue = new LinkedList<String>();
        queue.offer(beginWord);
        HashMap<String, Integer> maps = new HashMap<String, Integer>(); //store the level of each string
        maps.put(beginWord, 1);
        if (wordList.contains(beginWord)) wordList.remove(beginWord);
        
        while (!queue.isEmpty()) {
            String top = queue.poll();
            int len = top.length();
            StringBuilder builder;
            
            int level = maps.get(top);
            for (int i = 0; i < len; i++) {
                //find the strings which is one char diff with top
                builder = new StringBuilder(top);
                for (char c = 'a'; c <= 'z'; c++) {
                    builder.setCharAt(i, c);
                    String tmpStr = builder.toString();
                    if (tmpStr.equals(top))//match top
                        continue;
                    //add to next level
                    if (wordList.contains(tmpStr)) {
                        if (tmpStr.equals(endWord))//match endWord->return 
                            return level+1;
                        queue.offer(tmpStr);
                        wordList.remove(tmpStr);
                        maps.put(tmpStr, level+1);
                    }
                }
            }
        }
        return 0;
    }
}
```

Solution 2 Bidirectional Breadth First Search
```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if(!wordList.contains(endWord)) return 0;
        //top->down
        Queue<String> queue1 = new LinkedList<>();
        queue1.add(beginWord);
        //down->top
        Queue<String> queue2 = new LinkedList<>();
        queue2.add(endWord);
        
        Set<String> visited = new HashSet<>();
        visited.add(endWord);
        
        int step = 1;
        while(queue1.size() > 0 && queue2.size() > 0) {
            // always start from smaller number of queue 
            if(queue1.size() > queue2.size()) {
                Queue<String> temp = queue1;
                queue1 = queue2;
                queue2 = temp;
            }
            
            Queue<String> nextQueue = new LinkedList<>();
            while(!queue1.isEmpty()) {
                String cur = queue1.poll();
                for(String word: wordList) {
                    if(valid(cur, word)) {
                        if(queue2.contains(word)) {
                            return step+1;
                        }
                        
                        if(!visited.contains(word)) {
                            nextQueue.add(word);
                            visited.add(word);                            
                        }
                    }
                }
            }
            queue1 = nextQueue;
            step++;
        }
        return 0;
    }
    //whether step==1
    boolean valid(String a, String b) {
        int diff = 0;
        for(int i = 0; i < a.length(); ++i) {
            if(a.charAt(i) != b.charAt(i)) {
                diff++;
                if(diff >= 2) {
                    return false;
                }
            }
        }
        return true;
    }
}
```

## DP (DP<-->DFS + memo)

### 53.Maximum Subarray (Easy) [@](https://leetcode.com/problems/maximum-subarray/)

> Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
>
> **Example:**
>
> ```
> Input: [-2,1,-3,4,-1,2,1,-5,4],
> Output: 6
> Explanation: [4,-1,2,1] has the largest sum = 6.
> ```
>
> **Follow up:**
>
> If you have figured out the O(*n*) solution, try coding another solution using the divide and conquer approach, which is more subtle.

**Solution 1**

- 遍历所有子序列O(n^3) -> 住需要遍历从起始位置开始的子序列 O(n^2) ->
- 起始位置为负时，显然不是最大子序列和起始点。所以从负数部位最大子序列和的起点出发 O(n)
```java
class Solution {
    public int maxSubArray(int[] nums) {
        int nSize = nums.length;
        if (nSize == 0) return 0;
        int maxSum = Integer.MIN_VALUE;
        int nSum = 0;
        for (int i = 0; i < nSize; i++) {
            nSum += nums[i];
            if (nSum > maxSum)
                maxSum = nSum;
            if (nSum < 0)
                nSum = 0;
        }
        return maxSum;
    }
}
```

### 62. Unique Paths (Medium) [@](https://leetcode.com/problems/unique-paths/)

> A robot is located at the top-left corner of a *m* x *n* grid (marked 'Start' in the diagram below).
>
> The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
>
> How many possible unique paths are there?
>
> **Note:** *m* and *n* will be at most 100.
>
> **Example 1:**
>
> ```
> Input: m = 3, n = 2
> Output: 3
> Explanation:
> From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
> 1. Right -> Right -> Down
> 2. Right -> Down -> Right
> 3. Down -> Right -> Right
> ```

**Solution 1** 

- dp[i] [j] = dp[i-1] [j] + dp[i] [j-1]
- O(m*n)
```java
class Solution {
    public int uniquePaths(int m, int n) {
        //dp[i][j] = dp[i-1][j] + dp[i][j-1];
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i > 0)
                    dp[i][j] += dp[i-1][j];
                if (j > 0)
                    dp[i][j] += dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
}
```

**Solution 2**

- 空间复杂度 O(m*n) -> O(n)
- dp[j]: (0,0) -> (i,j)
- dp[j-1]表示dp[j]上方的值
- dp[j] = dp[j] + dp[j-1]
- 一列一列更新，只保存一列的数据
```java
class Solution {
    public int uniquePaths(int m, int n) {
        //dp[j]: num of paths from (0,0) to (i-1,j)
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j-1];
            }
        }
        return dp[n-1];
    }
}
```

### 70. Climbing Stairs (Easy) [@](https://leetcode.com/problems/climbing-stairs/)

> You are climbing a stair case. It takes *n* steps to reach to the top.
>
> Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
>
> **Note:** Given *n* will be a positive integer.
>
> **Example 1:**
>
> ```
> Input: 2
> Output: 2
> Explanation: There are two ways to climb to the top.
> 1. 1 step + 1 step
> 2. 2 steps
> ```
>
> **Example 2:**
>
> ```
> Input: 3
> Output: 3
> Explanation: There are three ways to climb to the top.
> 1. 1 step + 1 step + 1 step
> 2. 1 step + 2 steps
> 3. 2 steps + 1 step
> ```

Solution 1 DP
- 假设梯子有n层，那么如何爬到第n层呢，因为每次只能怕1或2步，那么爬到第n层的方法要么是从第n-1层一步上来的，要不就是从n-2层2步上来的，所以递推公式非常容易的就得出了
```java
class Solution {
    public int climbStairs(int n) {
        //dp[i] = dp[i-1] + dp[i-2];
        if (n == 1) return 1;
        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```
Solution 2 Fibonacci Number
```java
class Solution {
    public int climbStairs(int n) {
        if (n == 1) return 1;
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }
}
```

### 91. Decode Ways (Medium) [@](https://leetcode.com/problems/decode-ways/)

> A message containing letters from `A-Z` is being encoded to numbers using the following mapping:
>
> ```
> 'A' -> 1
> 'B' -> 2
> ...
> 'Z' -> 26
> ```
>
> Given a **non-empty** string containing only digits, determine the total number of ways to decode it.
>
> **Example 1:**
>
> ```
> Input: "12"
> Output: 2
> Explanation: It could be decoded as "AB" (1 2) or "L" (12).
> ```
>
> **Example 2:**
>
> ```
> Input: "226"
> Output: 3
> Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
> ```

**Solution**

- 设定状态为：`dp[i]`表示`s`从`0`开始，长度为`i`的子串的解码方式数量，于是我们最终要求的答案便是`dp[n]`。

  那么如何求解`dp[i]`呢？这个很简单，枚举最后一个字母对应1位还是2位，将f转化为规模更小的子问题。

  - 设`dp[i] = 0`
  - 枚举最后一个字母对应1位（要求`s[i - 1] != '0'`)，那么有`dp[i] += dp[i-1]`；
  - 枚举最后一个字母对应2位（要求`i > 1`且`s[i - 2]`和`s[i - 1]`组成的字符串在`"10"~"26"`的范围内），那么有`dp[i] += dp[i - 2]`；

- 也就是说，我们可以通过dp[i - 1]和dp[i - 2]计算出dp[i]来，这就是我们的状态和转移方程。

- 在具体实现中，我们可以按照i从1到n的顺序，依次计算出所有的dp[i]。
```java
class Solution {
    public int numDecodings(String s) {
        if (s.length() == 0) return 0;
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        //dp[i] 表示s从0开始，长度为i的字串的解码方式数量
        for (int i = 1; i < s.length()+1; i++) {
            if (s.charAt(i-1) != '0') 
                dp[i] += dp[i-1];
            if (i >= 2 && (s.substring(i-2, i).compareTo("10") >= 0 && s.substring(i-2, i).compareTo("26") <= 0))
                dp[i] += dp[i - 2];
        }
        return dp[s.length()];
    }
}
```

### 121. Best Time to Buy and Sell Stock (Easy) [@](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

>  

**Solution**

如果是动态规划的思路， 基本上我们要定义状态dp[i]， 然后看dp[i]和dp[i-1]或者dp[i-k]之间的关系。
假设我们定义dp[i]是在i天的最大利润， 那么和前面的重叠子问题的关系是什么呢？

- 一种情况当然是前面子问题里面的最大利润已经是整体的最大利润， 那么dp[i]=dp[i-1]
  还有一种情况是， 前面虽然取得了利润， 但是第i天卖出（对应到前面某一天买入)会产生更大的利润
  这时候，dp[i] = prices[i] - prices[j]
  也就是说, 整个递推公式是: dp[i] = Math.max(dp[i-1], prices[i]-prices[j]), 其中, j<i
  这样， 对于每个dp[i], 都只和之前的状态和数据有关， 和后面的选择已经无关了。
  然后这时候要考虑， prices[j]是哪个值会产生最大利润？ 当然是目前为止的最小值。
  也就是说， dp[j] = min prices so far， 而且这个值的好处是， 在一次遍历的过程中，可以直接随着遍历更新这个值。那么， 可以保存一个min值， 这样整体一次遍历就可以了。
- 有一个错误的思路， 就是一次遍历求出最小价格和最大价格， 然后得出利润。
  这个解法的错误的地方在于， 最大价格可能是最小价格的前面， 不能直接使用。
  反例比如[3,1,2]
- 前面的错误在于把顺序不符合要求的情况包括进去了，
  当然， 这个过程可以更简化。 甚至可以不需要用这么复杂的动态规划的思路, 直接对问题进行分析。
  对于最大利润的买入和卖出位置， 虽然买入和卖出可能出现在任意位置， 但是我们考虑如果固定其中一个价格会怎么样？
  实际上， 如果买入的位置已经选中， 那么卖出的位置也确定了。 反过来也成立， 如果卖出的位置已经选择， 那么买入的位置也确定了。

这里假设卖出的位置是i, 那么， 买入的位置就是在i前面的价格里面的最小价格。
那么，如果我们从左向右遍历， 每次保存目前已经遇到过的最小价格， 那么，prices[i]-min就是在i这个位置卖出的最大利润，这样就可以在一次遍历的过程中求解整体的最大利润。

```
class Solution {
    public int maxProfit(int[] prices) {
        if (prices.length == 0) return 0;
        int[] dp = new int[prices.length];
        int min = prices[0];
        
        for (int i = 1; i < prices.length; i++) {
            min = Math.min(prices[i], min);
            dp[i] = Math.max(dp[i-1], prices[i] - min);
        }
        return dp[prices.length - 1];
    }
}
```
- 可将dp[]换为max，降低空间复杂度

### 139. Word Break (Medium) [@](https://leetcode.com/problems/word-break/)

Solution 1

1. 一个直观的思路是暴力解，首先从头开始，看看每个单词能不能成为成为字符串的开头， 如果匹配上了， 可以对后面的继续这个过程
2. 但是这个过程有一点重复， 其实每次计算都是计算的时候，问题是判断某一个子字符串是不是满足要求， 而某一个子字符串，在这个问题里面其实就是原始字符串的index， 那么， 这个子问题可能是重叠的。
   比如， 针对"abcdef"和[“ab”, “cd”, “abcd”]
   那么， 针对index=4 （从1开始计数， 可以有ab+cd 或者abcd两种方式， 那么，一个计算过了，后面的就不需要再计算了。
3. 这样，就可以应用动态规划的思想， 设置dp[i]表示在i位已经满足要求的， 然后从前向后遍历，看看每一位是否可以走到更多的位；
4. 动态规划的常用套路，就是看prefix， 因为计算prefix的时候，问题已经求解过了，固定了； 当然要从postfix去理解也可以， 但是那样通常会是解问题的自然思路，但是从动态规划bottom up的方式，往往不是那么好理解。
   而当然，如果用记忆化递归的方式去理解，也是可以的。 但是同样要抽象出需要记忆的状态。 对于每个substring， 其实也是要用index来定义状态。 当然，完全用string做key也可能可以， 但是那样会浪费很多空间。

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        //i--开始位置
        for (int i = 0; i < s.length(); i++) {
            if (!dp[i]) continue;
            //j--结束位置
            for (int j = i+1; j <= s.length(); j++) {
                String subStr = s.substring(i, j);
                if (wordSet.contains(subStr)) {
                    dp[j] = true;
                }
            }
        }
        return dp[s.length()];
    }
}
```

### 140. Word Break II (Hard) [@](https://leetcode.com/problems/word-break-ii/)
> Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words, add spaces in *s* to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.
>
> **Note:**
>
> - The same word in the dictionary may be reused multiple times in the segmentation.
> - You may assume the dictionary does not contain duplicate words.
>
> **Example 1:**
>
> ```
> Input:
> s = "catsanddog"
> wordDict = ["cat", "cats", "and", "sand", "dog"]
> Output:
> [
>   "cats and dog",
>   "cat sand dog"
> ]
> ```
>
> **Example 2:**
>
> ```
> Input:
> s = "pineapplepenapple"
> wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
> Output:
> [
>   "pine apple pen apple",
>   "pineapple pen apple",
>   "pine applepen apple"
> ]
> Explanation: Note that you are allowed to reuse a dictionary word.
> ```
>
> **Example 3:**
>
> ```
> Input:
> s = "catsandog"
> wordDict = ["cats", "dog", "sand", "and", "cat"]
> Output:
> []
> ```

Solution Recursion

- Python
- 递归调用wordBerak()
- [Youtube 题解](https://www.youtube.com/watch?v=JqOIRBC0_9c)
- ![leetcode_140.png](:/ce516b8d2d2e441d9d46426147b9d38c)

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        words = set(wordDict)
        memo = {}
        def wordBreak(s):
            # already in memory, return directly
            if s in memo: 
                return memo[s]
            # answer for s
            ans = []
            if s in words:
                ans.append(s)
            for i in range(1, len(s)):
                # check whether right part is a word
                right = s[i:]
                if right not in words:
                    continue
                # append to the answer for left part
                ans += [w + " " + right for w in wordBreak(s[0:i])]
            memo[s] = ans
            return memo[s]
        return wordBreak(s)
```

### 152. Maximum Product Subarray (Medium) [@](https://leetcode.com/problems/maximum-product-subarray/)

> Given an integer array `nums`, find the contiguous subarray within an array (containing at least one number) which has the largest product.
>
> **Example 1:**
>
> ```
> Input: [2,3,-2,4]
> Output: 6
> Explanation: [2,3] has the largest product 6.
> ```
>
> **Example 2:**
>
> ```
> Input: [-2,0,-1]
> Output: 0
> Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
> ```

Solution DP

- 同时记录最大积和最小积，dp[i][0]表示以nums[i]结尾的子序列的最小积，dp[i][1]表示以nums[i]结尾的子序列的最大积。初始状态：
  dp[0] [0] = nums[0];
  dp[0] [1] = nums[0];
- 由于可能存在负数，所以有三个数参与判断，状态转移方程：
  dp[i] [0] = min( min(dp[i - 1] [0] * nums[i], dp[i - 1] [1] * nums[i]), nums[i])
  dp[i] [1] = max( max(dp[i - 1] [0] * nums[i], dp[i - 1] [1] * nums[i]), nums[i])
- 可以在用一个变量result记录结果，每次计算出最大积时就更新一下result，最后返回result就行，见下面我的代码1，时间复杂度是O(n)O(n)，空间复杂度是O(n)O(n)
- 通过状态转移方程可以看出计算dp[i] []时只需要用到dp[i - 1] []，与dp[i - 2] []及前面的结果没有关系，因此空间复杂度可以进一步优化，只用两个变量localMin和localMax存储前一个位置的最大积和最小积

```java
class Solution {
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int localMin = nums[0];
        int localMax = nums[0];
        int globalMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int tmp = localMin;
            localMin = Math.min(Math.min(tmp * nums[i], localMax * nums[i]), nums[i]);
            localMax = Math.max(Math.max(localMax * nums[i], tmp * nums[i]), nums[i]);
            globalMax = Math.max(localMax, globalMax);
        }
        return globalMax;
    }
}
```

### 198.House Robber (Easy) [@](https://leetcode.com/problems/house-robber/)

> You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.
>
> Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight **without alerting the police**.
>
> **Example 1:**
>
> ```
> Input: [1,2,3,1]
> Output: 4
> Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
>           Total amount you can rob = 1 + 3 = 4.
> ```
>
> **Example 2:**
>
> ```
> Input: [2,7,9,3,1]
> Output: 12
> Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
>           Total amount you can rob = 2 + 9 + 1 = 12.
> ```

Solution 1 DP

- 递推公式

~~~
dp[0] = num[0] （当i=0时）
dp[1] = max(num[0], num[1]) （当i=1时）
dp[i] = max(num[i] + dp[i - 2], dp[i - 1])   （当i !=0 and i != 1时）
~~~

```java
class Solution {
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) 
            return 0;
        int[] dp = new int[nums.length+1];
        for (int i = 0; i < nums.length; i++) {
            if (i == 0)
                dp[i] = nums[i];
            else if (i == 1)
                dp[i] = Math.max(nums[i], nums[i-1]);
            else
                dp[i] = Math.max(dp[i-2]+nums[i], dp[i-1]);
        }
        return dp[nums.length-1];
    }
}
```
Solution 2

- 优化空间复杂度 O(1)

```java
    public int rob(int[] nums) {
        int rob = 0, notrob = 0;
        for (int i = 0; i < nums.length; i++) {
            int temp = rob;
            rob = notrob + nums[i];
            notrob = Math.max(temp, notrob);
        }
        return Math.max(rob, notrob);
    }
```

## Binary Search

### 69. Sqrt(x) (Easy) [@](https://leetcode.com/problems/sqrtx/)

> Implement `int sqrt(int x)`.
>
> Compute and return the square root of *x*, where *x* is guaranteed to be a non-negative integer.
>
> Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.
>
> **Example 1:**
>
> ```
> Input: 4
> Output: 2
> ```
>
> **Example 2:**
>
> ```
> Input: 8
> Output: 2
> Explanation: The square root of 8 is 2.82842..., and since 
>           the decimal part is truncated, 2 is returned.
> ```

```
class Solution {
    public int mySqrt(int x) {
        if (x < 2) return x;
        int left = 0;
        int right = x/2+1;
        long mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (mid * mid == x) {
                return (int)mid;
            }else if (mid * mid > x) {
                right = (int)mid -1;
            }else {
                left = (int)mid + 1;
            }
        }
        return right;
    }
}
```
### 162. Find Peak Element (Medium) [@](https://leetcode.com/problems/find-peak-element/)

> A peak element is an element that is greater than its neighbors.
>
> Given an input array `nums`, where `nums[i] ≠ nums[i+1]`, find a peak element and return its index.
>
> The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
>
> You may imagine that `nums[-1] = nums[n] = -∞`.
>
> **Example 1:**
>
> ```
> Input: nums = [1,2,3,1]
> Output: 2
> Explanation: 3 is a peak element and your function should return the index number 2.
> ```
>
> **Example 2:**
>
> ```
> Input: nums = [1,2,1,3,5,6,4]
> Output: 1 or 5 
> Explanation: Your function can return either index number 1 where the peak element is 2, 
>           or index number 5 where the peak element is 6.
> ```

Solution
- 因为nums[-1] = nums[n] = -∞, 所以当nums[mid] < nums[mid+1] 时，mid右侧必定有peak，同理点那个nums[mid] >= nums[mid+1]时，mid及其左侧必有peak
```
class Solution {
    public int findPeakElement(int[] nums) {
        int n = nums.length;
        int left = 0, right = n-1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (nums[mid] < nums[mid+1]) {
                //mid右侧必定有peak
                left = mid + 1;
            }else {
                //包括mid在内左侧必有peak
                right = mid;
            }
        }
        return left;
    }
}
```

## Greedy

### 55. Jump Game (Medium) [@](https://leetcode.com/problems/jump-game/)

> Given an array of non-negative integers, you are initially positioned at the first index of the array.
>
> Each element in the array represents your maximum jump length at that position.
>
> Determine if you are able to reach the last index.
>
> **Example 1:**
>
> ```
> Input: [2,3,1,1,4]
> Output: true
> Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
> ```
>
> **Example 2:**
>
> ```
> Input: [3,2,1,0,4]
> Output: false
> Explanation: You will always arrive at index 3 no matter what. Its maximum
>           jump length is 0, which makes it impossible to reach the last index.
> ```

Solution 1 Greedy
- 维护一个reach（最远可达距离），每次前进一步，如果i一直在reach范围内，则可达
```
class Solution {
    public boolean canJump(int[] nums) {
        int reach = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > reach)
                return false;
            reach = Math.max(reach, i + nums[i]);
        }
        return true;
    }
}
```

Solution 2 Zero Point
- 若无0点则一定可达任一点
- 故只需考虑0点，判断可否跳过此0点即此0点向前数第k个位置的元素大于k即可跳过
```
class Solution {
    public boolean canJump(int[] nums) {
        int i = nums.length - 2; //0点
        while(i >= 0) {
            if (nums[i] == 0) {
                int j = i - 1;//向前找可以跳过0点的位置
                while (j >= 0) {
                    if (j + nums[j] > i) {
                        break;
                    }
                    j--;
                }
                if (j == -1)
                    return false;
            }
            i--;
        }
        return true;
    }
}
```

### 122. Best Time to Buy and Sell Stock II (Easy) [@](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
> Say you have an array for which the *i*th element is the price of a given stock on day *i*.
>
> Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).
>
> **Note:** You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
>
> **Example 1:**
>
> ```
> Input: [7,1,5,3,6,4]
> Output: 7
> Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
>           Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
> ```
>
> **Example 2:**
>
> ```
> Input: [1,2,3,4,5]
> Output: 4
> Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
>           Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
>           engaging multiple transactions at the same time. You must sell before buying again.
> ```
>
> **Example 3:**
>
> ```
> Input: [7,6,4,3,1]
> Output: 0
> Explanation: In this case, no transaction is done, i.e. max profit = 0.
> ```

Solution Greedy

- 累计所有前低后高的差值
```
class Solution {
    public int maxProfit(int[] prices) {
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] - prices[i-1] > 0)
                profit += prices[i] - prices[i-1];
        }
        return profit;
    }
}
```

### 134. Gas Station (Medium) [@](https://leetcode.com/problems/gas-station/)

> There are *N* gas stations along a circular route, where the amount of gas at station *i* is `gas[i]`.
>
> You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from station *i* to its next station (*i*+1). You begin the journey with an empty tank at one of the gas stations.
>
> Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.
>
> **Note:**
>
> - If there exists a solution, it is guaranteed to be unique.
> - Both input arrays are non-empty and have the same length.
> - Each element in the input arrays is a non-negative integer.
>
> **Example 1:**
>
> ```
> Input: 
> gas  = [1,2,3,4,5]
> cost = [3,4,5,1,2]
> 
> Output: 3
> 
> Explanation:
> Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
> Travel to station 4. Your tank = 4 - 1 + 5 = 8
> Travel to station 0. Your tank = 8 - 2 + 1 = 7
> Travel to station 1. Your tank = 7 - 3 + 2 = 6
> Travel to station 2. Your tank = 6 - 4 + 3 = 5
> Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
> Therefore, return 3 as the starting index.
> ```
>
> **Example 2:**
>
> ```
> Input: 
> gas  = [2,3,4]
> cost = [3,4,3]
> 
> Output: -1
> 
> Explanation:
> You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
> Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
> Travel to station 0. Your tank = 4 - 3 + 2 = 3
> Travel to station 1. Your tank = 3 - 3 + 3 = 3
> You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
> Therefore, you can't travel around the circuit once no matter where you start.
> ```

Solution Greedy
- sum(gas) >= sum(cost) => 有解
- 只要找到一个起点i，从这个点出发的所有gas的和总比cost和打即可
```
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum = 0, subsum = 0, begin = 0;
        for (int i = 0; i < gas.length; i++) {
            sum += gas[i] - cost[i];
            subsum += gas[i] - cost[i];
            if (subsum < 0) {
                subsum = 0;
                begin = i + 1;
            }
        }
        if (sum < 0) return -1;
        return begin;
    }
}
```
## Tree

### 94. Binary Tree Inorder Traversal (Medium)  [@](https://leetcode.com/problems/binary-tree-inorder-traversal/)

> Given a binary tree, return the *inorder* traversal of its nodes' values.
>
> **Example:**
>
> ```
> Input: [1,null,2,3]
> 1
>  \
>   2
>  /
> 3
> 
> Output: [1,3,2]
> ```
>
> **Follow up:** Recursive solution is trivial, could you do it iteratively?

Solution 1 Recursion

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        helper(root, res);
        return res;
    }
    private void helper(TreeNode root, List<Integer> res) {
        if (root != null) {
            if (root.left != null) {
                helper(root.left, res);
            }
            res.add(root.val);
            if (root.right != null) {
                helper(root.right, res);
            }
        }
    }
}
```

Solution 2 Stack

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            res.add(cur.val);
            cur = cur.right;
        }
        return res;
    }
}
```

## Backtracking

### 78. Subsets (Medium) [@](https://leetcode.com/problems/subsets/)

> Given a set of **distinct** integers, *nums*, return all possible subsets (the power set).
>
> **Note:** The solution set must not contain duplicate subsets.
>
> **Example:**
>
> ```
> Input: nums = [1,2,3]
> Output:
> [
> [3],
> [1],
> [2],
> [1,2,3],
> [1,3],
> [2,3],
> [1,2],
> []
> ]
> ```

Solution 1 Recursion
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> cur = new ArrayList<Integer>();
        backtrack(res, cur, nums, 0);
        return res;
    }
    private void backtrack(List<List<Integer>> res, List<Integer> cur, int[] nums, int j) {
        res.add(new ArrayList<Integer>(cur));
        for (int i = j; i < nums.length; i++) {
            cur.add(nums[i]);//add nums[i]
            backtrack(res, cur, nums, i+1);// Recursion
            cur.remove(cur.size()-1);//remove nums[i]
        }
    }
}
```
Solution 2 Iteration
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        res.add(new ArrayList<Integer>());
        for (int num : nums) { //pick up each element from nums
            int size = res.size();
            for (int i = 0; i < size; i++) {
                //pick up each element in current res
                List<Integer> temp = new ArrayList<Integer>(res.get(i));
                temp.add(num);//put num into temp
                res.add(temp);//add temp into res
            }
        }
        return res;
    }
}
```

### 79. Word Search (Medium) [@](https://leetcode.com/problems/word-search/)

> Given a 2D board and a word, find if the word exists in the grid.
>
> The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.
>
> **Example:**
>
> ```
> board =
> [
> ['A','B','C','E'],
> ['S','F','C','S'],
> ['A','D','E','E']
> ]
> 
> Given word = "ABCCED", return true.
> Given word = "SEE", return true.
> Given word = "ABCB", return false.
> ```

Solution dfs + backtrack
```java
class Solution {
    //direction: right, down, left, up
    int[] drow = {0, 1, 0, -1};
    int[] dcol = {1, 0, -1, 0};
    public boolean exist(char[][] board, String word) {
        boolean[][] isVisited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (isThisWay(board, word, i, j, 0, isVisited))
                    return true;
            }
        }
        return false;
    }
    
    private boolean isThisWay(char[][] board, String word, int row, int col, int index, boolean[][] isVisited) {
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length || isVisited[row][col] || board[row][col] != word.charAt(index))
            return false;
        if (++index == word.length())
            return true; // complete matching
        isVisited[row][col] = true;
        for (int i = 0; i < 4; i++) {
            if (isThisWay(board, word, row + drow[i], col + dcol[i], index, isVisited))
                return true;
        }
        isVisited[row][col] = false;//backtrack if false
        return false;
    }
}
```

### 131. Palindrome Partitioning (Medium) [@](https://leetcode.com/problems/palindrome-partitioning/)

Solution DFS + backtracking
- 递归寻找子问题，如果子串回文，则加入res
```java
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<List<String>>();
        List<String> cur = new ArrayList<String>();
        if (s.length() == 0 || s == null) return res;
        
        backtrack(s, 0, cur, res);
        return res;
    }
    
    private void backtrack(String s, int start, List<String> cur, List<List<String>> res) {
        //recursion complete condition
        if (start == s.length()) {
            res.add(new ArrayList<String>(cur));
            return;
        }
        for (int i = start; i < s.length(); i++) {
            String str = s.substring(start, i + 1);
            if (isPalindrome(str)) {
                cur.add(str);
                backtrack(s, i+1, cur, res);
                cur.remove(cur.size()-1);
            }
        }
    }
    
    private boolean isPalindrome(String str) {
        int left = 0, right = str.length() - 1;
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
```
### 212. Word Search II (Hard) [@](https://leetcode.com/problems/word-search-ii/)

> Given a 2D board and a list of words from the dictionary, find all words in the board.
>
> Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
>
> 
>
> **Example:**
>
> ```
> Input: 
> board = [
> ['o','a','a','n'],
> ['e','t','a','e'],
> ['i','h','k','r'],
> ['i','f','l','v']
> ]
> words = ["oath","pea","eat","rain"]
> 
> Output: ["eat","oath"]
> ```
>
> 
>
> **Note:**
>
> 1. All inputs are consist of lowercase letters `a-z`.
> 2. The values of `words` are distinct.

[Solution](https://www.cnblogs.com/Dylan-Java-NYC/p/4944555.html) Tire + DFS

> [Word Search](http://www.cnblogs.com/Dylan-Java-NYC/p/4944270.html)的进阶版题目，同时可以利用[Implement Trie (Prefix Tree)](http://www.cnblogs.com/Dylan-Java-NYC/p/4888830.html).
>
> 生成Trie树，把所有的词都insert进去。
>
> 然后从board上的每一个char开始dfs查找。
>
> 终止条件有两个， 一个 i 和 j 出界，或者board[i][j]已经用过了. 另一个是把board[i][j]加到当前item后，若没有以更新过item为prefix的时候就可以返回了.
>
> search 更新过的item, 若是有就加到res中, **并且继续，这里不能return,** 因为有可能有 "aabc" "aabcb"两个词同时存在的情况，只检查了"aabc"就return会漏掉"aabcb".
>
> 标记当前used为true, 然后board四个方向都做recursion. used再改回来.
>
> Note: 如果board 是[a a], words 只有一个[a], 此时小心重复加了，所以要用HashSet生成res, 最后再用res生成的List返回。
>
> m = board.length, n = board[0].length, k = words.length, l 是 word的平均长度.
>
> Time Complexity: O(k*l + m*n*l*4^l). k*l是简历Trie用时间. m*n是外部循环, l是search Trie时间, 4^l是recursion + backtracking的时间.
>
> Space: O(k*l + l). k*l是Trie数的大小. 用了l层stack.

```java
public class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        HashSet<String> res = new HashSet<String>();
        if(words == null || words.length == 0 || board == null || board.length == 0 || board[0].length == 0){
            return new ArrayList(res);
        }
        Trie trie = new Trie();
        for(int i = 0; i<words.length; i++){
            trie.insert(words[i]);
        }
        
        boolean [][] used = new boolean[board.length][board[0].length];
        for(int i = 0; i<board.length; i++){
            for(int j = 0; j<board[0].length; j++){
                findHelper(board,trie,used,"",i,j,res);
            }
        }
        return new ArrayList(res);
    }
    private void findHelper(char[][] board, Trie trie, boolean [][] used, String item, int i, int j, HashSet<String> res){
        
        if(i<0 || j<0 || i>= board.length || j>=board[0].length || used[i][j]){
            return;
        }
        
        item = item+board[i][j];
        if(!trie.startsWith(item)){
            return;
        }
        if(trie.search(item)){
            res.add(item);
        }
        used[i][j] = true;
        findHelper(board,trie,used,item,i+1,j,res);
        findHelper(board,trie,used,item,i-1,j,res);
        findHelper(board,trie,used,item,i,j+1,res);
        findHelper(board,trie,used,item,i,j-1,res);
        used[i][j] = false;
    }
}


class TrieNode{
    String val = "";
    TrieNode [] nexts;
    public TrieNode(){
        nexts = new TrieNode[26];
    }
}
class Trie{
    private TrieNode root;
    public Trie(){
        root = new TrieNode();
    }
    
    public void insert(String word){
        TrieNode p = root;
        for(char c : word.toCharArray()){
            if(p.nexts[c-'a'] == null){
                p.nexts[c-'a'] = new TrieNode();
            }
            p = p.nexts[c-'a'];
        }
        p.val = word;
    }
    
    public boolean search(String word){
        TrieNode p = root;
        for(char c : word.toCharArray()){
            if(p.nexts[c-'a'] == null){
                return false;
            }
            p = p.nexts[c-'a'];
        }
        return p.val.equals(word);
    }
    
    public boolean startsWith(String prefix){
        TrieNode p = root;
        for(char c : prefix.toCharArray()){
            if(p.nexts[c-'a'] == null){
                return false;
            }
            p = p.nexts[c-'a'];
        }
        return true;
    }
}
```
## String

### 12. Integer to Roman (Easy) [@](https://leetcode.com/problems/integer-to-roman/)

> Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.
>
> ```
> Symbol       Value
> I             1
> V             5
> X             10
> L             50
> C             100
> D             500
> M             1000
> ```
>
> For example, two is written as `II` in Roman numeral, just two one's added together. Twelve is written as, `XII`, which is simply `X` + `II`. The number twenty seven is written as `XXVII`, which is `XX` + `V` + `II`.
>
> Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:
>
> - `I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
> - `X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
> - `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.
>
> Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.
>
> **Example 1:**
>
> ```
> Input: 3
> Output: "III"
> ```
>
> **Example 2:**
>
> ```
> Input: 4
> Output: "IV"
> ```
>
> **Example 3:**
>
> ```
> Input: 9
> Output: "IX"
> ```
>
> **Example 4:**
>
> ```
> Input: 58
> Output: "LVIII"
> Explanation: L = 50, V = 5, III = 3.
> ```

Solution

- 计算每个位的值，并用对应字符串表示

```java
class Solution {
    public String intToRoman(int num) {
        String[] M = {"", "M", "MM", "MMM"};
        String[] C = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] X = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] I = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        
        return M[(num/1000)]+C[(num%1000)/100]+X[(num%100)/10]+I[(num%10)];
    }
}
```



### 13. Roman to Integer (Easy) [@](https://leetcode.com/problems/roman-to-integer/)

> Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.
>
> ```
> Symbol       Value
> I             1
> V             5
> X             10
> L             50
> C             100
> D             500
> M             1000
> ```
>
> For example, two is written as `II` in Roman numeral, just two one's added together. Twelve is written as, `XII`, which is simply `X` + `II`. The number twenty seven is written as `XXVII`, which is `XX` + `V` + `II`.
>
> Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:
>
> - `I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
> - `X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
> - `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.
>
> Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.
>
> **Example 1:**
>
> ```
> Input: "III"
> Output: 3
> ```
>
> **Example 2:**
>
> ```
> Input: "IV"
> Output: 4
> ```
>
> **Example 3:**
>
> ```
> Input: "IX"
> Output: 9
> ```
>
> **Example 4:**
>
> ```
> Input: "LVIII"
> Output: 58
> Explanation: L = 50, V= 5, III = 3.
> ```
>
> **Example 5:**
>
> ```
> Input: "MCMXCIV"
> Output: 1994
> Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
> ```

Solution 1

```java
class Solution {
    public int romanToInt(String s) {
        HashMap<Character, Integer> hm = new HashMap<Character, Integer>();
        hm.put('I', 1);
        hm.put('V', 5);
        hm.put('X', 10);
        hm.put('L', 50);
        hm.put('C', 100);
        hm.put('D', 500);
        hm.put('M', 1000);
        
        int maxDigit = 0, val = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            int cur = hm.get(s.charAt(i));
            if (cur >= maxDigit) {
                val += cur;
                maxDigit = cur;
            }else {
                val -= cur;
            }
        }
        return val;
    }
}
```

Solution 2

- 仅有 I X C 可能成为前缀，所以遇到需要考虑是否更新前缀

```java
class Solution {
    public static int romanToInt(String s) {
        int x = 0;
        char prev = ' ';
        for (int i = 0; i < s.length(); i++) {
            // if (prev == ' ') prev = s.charAt(i);
            
            switch (s.charAt(i)) {
                case 'M':
                    x += prev == 'C' ? 900 : 1000;
                    break;
                case 'D':
                    x += prev == 'C' ? 400 : 500;
                    break;
                case 'C':
                    if (i < s.length() - 1 && (s.charAt(i + 1) == 'D' || s.charAt(i + 1) == 'M')) {
                        prev = 'C';
                    } else {
                        x += prev == 'X' ? 90 : 100;
                    }
                    break;
                case 'L':
                    x += prev == 'X' ? 40 : 50;
                    break;
                case 'X':
                    if (i < s.length() - 1 && (s.charAt(i + 1) == 'L' || s.charAt(i + 1) == 'C')) {
                        prev = 'X';
                    } else {
                        x += prev == 'I' ? 9 : 10;
                    }
                    break;
                case 'V':
                    x += prev == 'I' ? 4 : 5;
                    break;
                case 'I':
                    if (i < s.length() - 1 && (s.charAt(i + 1) == 'V' || s.charAt(i + 1) == 'X')) {
                        prev = 'I';
                    } else {
                        x += 1;
                    }
                    break;
            }
        }
        return x;
    }
}
```



### 14. Longest Common Prefix (Easy) [@](https://leetcode.com/problems/longest-common-prefix/)

> Write a function to find the longest common prefix string amongst an array of strings.
>
> If there is no common prefix, return an empty string `""`.
>
> **Example 1:**
>
> ```
> Input: ["flower","flow","flight"]
> Output: "fl"
> ```
>
> **Example 2:**
>
> ```
> Input: ["dog","racecar","car"]
> Output: ""
> Explanation: There is no common prefix among the input strings.
> ```
>
> **Note:**
>
> All given inputs are in lowercase letters `a-z`.

Solution 1 Recursion + Divide and Conquer

- 二分所有串，一半一半考虑找出commonPrefix

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0 || strs == null) return ""; //特殊情况
        return longestCommonPrefix(strs, 0, strs.length-1);
    }
    private String longestCommonPrefix(String[] strs, int l, int r) {
        if (l == r) {
            return strs[l];
        }else {
            int mid = (l+r)/2;
            String lcpLeft = longestCommonPrefix(strs, l, mid);//左区间
            String lcpRight = longestCommonPrefix(strs, mid+1, r);//右区间
            return commonPrefix(lcpLeft, lcpRight);
        }
    }
    private String commonPrefix(String ls, String rs) {
        int min = Math.min(ls.length(), rs.length());
        for (int i = 0; i<min; i++) {
            if (ls.charAt(i) != rs.charAt(i)) {
                return ls.substring(0,i);
            }
        }
        return ls.substring(0,min);
    }
}
```

Solution 2 Divide and Conquer

- 假设commonPrefix长度，二分最短串长度
- 如果存在，则l和r最后汇聚在commonPrefix的尾部

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0 || strs == null) {
            return "";
        }
        
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < strs.length; i++) {
            minLen = Math.min(strs[i].length(), minLen);
        }
        int l = 0, r = minLen;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (isCommonPrefix(strs, mid)) {
                l = mid + 1;
            }else {
                r= mid - 1;
            }
        }
        return strs[0].substring(0, (l+r)/2);
    }
    private boolean isCommonPrefix(String[] strs, int len) {
        String s = strs[0].substring(0, len);
        for (int i = 1; i < strs.length; i++) {
            if (!strs[i].startsWith(s)) {
                return false;
            }
        }
        return true;
    }
}
```



### 28. Implement strStr()  (Easy)  [@](https://leetcode.com/problems/implement-strstr/)

> Implement [strStr()](http://www.cplusplus.com/reference/cstring/strstr/).
>
> Return the index of the first occurrence of needle in haystack, or **-1** if needle is not part of haystack.
>
> **Example 1:**
>
> ```
> Input: haystack = "hello", needle = "ll"
> Output: 2
> ```
>
> **Example 2:**
>
> ```
> Input: haystack = "aaaaa", needle = "bba"
> Output: -1
> ```

Solution 1 Iteration

```java
class Solution {
    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) return 0;
        for (int i = 0; i < haystack.length() - needle.length() + 1; i++) {
            if (haystack.charAt(i) == needle.charAt(0)) {
                int j = 1;
                while (j < needle.length() && haystack.charAt(i+j) == needle.charAt(j)) {
                    j++;
                }
                if (j == needle.length())
                    return i;
            }
        }
        return -1;
    }
}
```

Solution 2 KMP (Knuth–Morris–Pratt [string-searching algorithm](https://en.wikipedia.org/wiki/String-searching_algorithm))

- 核心是PMT(Partial Match Table)数组：前缀B -- A=BS (S为非空字符串)；PMT 中的值是字符串的前缀集合与后缀集合的交集中最长元素的长度
- ![IMG_0923.JPG](:/67dad347bcda4fb7998271ac90779b10)

```java
class Solution {
    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) return 0;
        int i = 0, j = -1, N = needle.length(), M = haystack.length();
        int[] next = new int[N];
        next[0] = -1;
        while (i < N-1) {// generate next array
            if (j == -1 || needle.charAt(i) == needle.charAt(j)) {
                i++;
                j++;
                next[i] = j;
            }else {
                j = next[j];
            }
        }
        
        i = 0; j = 0;
        while (i < M && j < N) {
            if (j == -1 || haystack.charAt(i) == needle.charAt(j)) {
                i++;
                j++;
            }else {
                j = next[j];
            }
        }
        if (j == N) 
            return i-j;
        return -1;
    }
}
```

Solution 3 HashMap

- 直接containsKey匹配子串

```java
class Solution {
    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) return 0;
        if (needle.length() > haystack.length()) return -1;
        
        HashMap<String, Integer> map = new HashMap<>();
        map.put(haystack, 0);
        //put each needle-len substring of haystack into the hashmap
        for (int i = 0; i <= haystack.length() - needle.length(); i++) {
            if (map.containsKey(needle)) {
                return map.get(needle);
            }
            map.put(haystack.substring(i, i+needle.length()), i);
        }
        return map.getOrDefault(needle, -1);
    }
}
```



### 38. Count and Say (Easy) [@](https://leetcode.com/problems/count-and-say/)

> The count-and-say sequence is the sequence of integers with the first five terms as following:
>
> ```
> 1.     1
> 2.     11
> 3.     21
> 4.     1211
> 5.     111221
> ```
>
> `1` is read off as `"one 1"` or `11`.
> `11` is read off as `"two 1s"` or `21`.
> `21` is read off as `"one 2`, then `one 1"` or `1211`.
>
> Given an integer *n* where 1 ≤ *n* ≤ 30, generate the *n*th term of the count-and-say sequence. You can do so recursively, in other words from the previous member read off the digits, counting the number of digits in groups of the same digit.
>
> Note: Each term of the sequence of integers will be represented as a string.
>
>  
>
> **Example 1:**
>
> ```
> Input: 1
> Output: "1"
> Explanation: This is the base case.
> ```
>
> **Example 2:**
>
> ```
> Input: 4
> Output: "1211"
> Explanation: For n = 3 the term was "21" in which we have two groups "2" and "1", "2" can be read as "12" which means frequency = 1 and value = 2, the same way "1" is read as "11", so the answer is the concatenation of "12" and "11" which is "1211".
> ```

Solution 1 Recursion

```java
class Solution {
    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }else {
            String preStr = countAndSay(n-1);
            String res = "";
            int len = 1; //len of same char
            int i = 1; //idx of preStr
            //scan preStr to determine the following string
            while (i < preStr.length()) {
                if (preStr.charAt(i) == preStr.charAt(i-1)) {
                    len++;
                }else {
                    res += String.valueOf(len) + String.valueOf(preStr.charAt(i-1));
                    len = 1; //reset len of same char
                }
                i++;
            }
            res += String.valueOf(len) + String.valueOf(preStr.charAt(i-1));
            return res.toString();
        }
    }
}
```

Solution 1 Improvement 

```java
class Solution {
    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }else {
            String preStr = countAndSay(n-1);
            StringBuilder res = new StringBuilder();
            int len = 1; //len of same char
            int i = 1; //idx of preStr
            //scan preStr to determine the following string
            while (i < preStr.length()) {
                if (preStr.charAt(i) == preStr.charAt(i-1)) {
                    len++;
                }else {
                    // res += String.valueOf(len) + String.valueOf(preStr.charAt(i-1));
                    res.append(len).append(preStr.charAt(i-1));
                    len = 1; //reset len of same char
                }
                i++;
            }
            // res += String.valueOf(len) + String.valueOf(preStr.charAt(i-1));
            res.append(len).append(preStr.charAt(i-1));
            return res.toString();
        }
    }
}
```



Solution 2 Recursion

```java
class Solution {
    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }
        return read(countAndSay(n-1));
    }
    private String read(String preStr) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < preStr.length(); i++) {
            char c = preStr.charAt(i); //current char
            int len = 1; //len of same char
            //find and append next "len+c"
            while ((i+1) < preStr.length()) {
                if (preStr.charAt(i+1) != c) {
                    break;
                } else {
                    i++;
                    len++;
                }
            }
            res.append(len).append(c);
        }
        return res.toString();
    }
}
```



### 49. Group Anagrams (Medium) [@](https://leetcode.com/problems/group-anagrams/)

> Given an array of strings, group anagrams together.
>
> **Example:**
>
> ```
> Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
> Output:
> [
> ["ate","eat","tea"],
> ["nat","tan"],
> ["bat"]
> ]
> ```
>
> **Note:**
>
> - All inputs will be in lowercase.
> - The order of your output does not matter.

Solution 1 Hash Map

- sort(str) 找到同字母的串

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        if(strs == null || strs.length == 0) {
            List<List<String>> ans = new ArrayList<List<String>>();
            return ans;
        }
        
        HashMap<String, List<String>> hash = new HashMap<String, List<String>>();
        for (String str:strs) {
            char[] c = str.toCharArray();
            Arrays.sort(c);
            String temp = String.valueOf(c);
            if (!hash.containsKey(temp)) {
                List<String> vals = new ArrayList<String>();
                vals.add(str);
                hash.put(temp, vals);
            }else {
                hash.get(temp).add(str);
            }
        }
        List<List<String>> ans = new ArrayList<List<String>>();
        ans.addAll(hash.values());
        return ans;
    }
}
```

Solution 2 Hash Map + Prime Number

- 每个字母对应一个质数，计算所有字符串的积

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        //26个质数对应26个字母
        int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103};
        List<List<String>> ans = new ArrayList<List<String>>();
        HashMap<Integer, List<String>> hash = new HashMap<Integer, List<String>>();
        for (String str:strs) {
            int key = 1;
            for (char c : str.toCharArray()) {
                key *= primes[c-'a'];
            }
            if (!hash.containsKey(key)) {
                List<String> vals = new ArrayList<String>();
                vals.add(str);
                hash.put(key, vals);
            } else {
                hash.get(key).add(str);
            }
        }
        ans.addAll(hash.values());
        return ans;
    }
}
```



### 58. Length of Last Word (Easy) [@](https://leetcode.com/problems/length-of-last-word/)

> Given a string *s* consists of upper/lower-case alphabets and empty space characters `' '`, return the length of last word (last word means the last appearing word if we loop from left to right) in the string.
>
> If the last word does not exist, return 0.
>
> **Note:** A word is defined as a **maximal substring** consisting of non-space characters only.
>
> **Example:**
>
> ```
> Input: "Hello World"
> Output: 5
> ```

Solution 

- Start = 第一个非' '字符
- End = 下一个‘ ’
- 注意while条件顺序，首先判断是否越界

```java
class Solution {
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0) return 0;
        
        int count = 0;
        int first = s.length() - 1;
        //the first not-' ' character --- start
        while (first >= 0 && s.charAt(first) == ' ')
            first--;
        //next ' ' position --- end
        while (first >= 0 && s.charAt(first) != ' ') {
            first--;
            count++;
        }
        return count;
    }
}
```



### 87.  Scramble String (Hard) [@](https://leetcode.com/problems/scramble-string/)

> Given a string *s1*, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.
>
> Below is one possible representation of *s1* = `"great"`:
>
> ```
>  great
> /    \
> gr    eat
> / \    /  \
> g   r  e   at
>         / \
>        a   t
> ```
>
> To scramble the string, we may choose any non-leaf node and swap its two children.
>
> For example, if we choose the node `"gr"` and swap its two children, it produces a scrambled string `"rgeat"`.
>
> ```
>  rgeat
> /    \
> rg    eat
> / \    /  \
> r   g  e   at
>         / \
>        a   t
> ```
>
> We say that `"rgeat"` is a scrambled string of `"great"`.
>
> Similarly, if we continue to swap the children of nodes `"eat"` and `"at"`, it produces a scrambled string `"rgtae"`.
>
> ```
>  rgtae
> /    \
> rg    tae
> / \    /  \
> r   g  ta  e
>     / \
>    t   a
> ```
>
> We say that `"rgtae"` is a scrambled string of `"great"`.
>
> Given two strings *s1* and *s2* of the same length, determine if *s2* is a scrambled string of *s1*.
>
> **Example 1:**
>
> ```
> Input: s1 = "great", s2 = "rgeat"
> Output: true
> ```
>
> **Example 2:**
>
> ```
> Input: s1 = "abcde", s2 = "caebd"
> Output: false
> ```

Solution 1 Recursion

```java
class Solution {
    public boolean isScramble(String s1, String s2) {
        //len of str
        if (s1.length() != s2.length()) return false;
        
        if (s1.equals(s2)) return true;
        
        //num of letters
        int[] letter = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            letter[s1.charAt(i) - 'a']++;
            letter[s2.charAt(i) - 'a']--;
        }
        //diff num of letters -> false
        for (int i = 0; i < 26; i++) {
            if (letter[i] != 0)
                return false;
        }
        
        //loop through all cut points
        for (int i = 1; i < s1.length(); i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i)))
                return true;
            //switch
            if (isScramble(s1.substring(i), s2.substring(0, s2.length() - i)) && isScramble(s1.substring(0, i), s2.substring(s2.length() - i)))
                return true;
        }
        return false;
   }
}
```

Solution 2 Recursion + Memorization

```java
class Solution {
    public boolean isScramble(String s1, String s2) {
        HashMap<String, Integer> memo = new HashMap<>();
        return isSrambleHelper(s1, s2, memo);
    }
    
    private boolean isSrambleHelper(String s1, String s2, HashMap<String, Integer> memo) {
        //previous res
        int res = memo.getOrDefault(s1 + "#" + s2, -1);
        if (res == 1)
            return true;
        else if (res == 0)
            return false;
        
        //len of str
        if (s1.length() != s2.length()) {
            memo.put(s1 + "#" + s2, 0);
            return false;
        } 
        
        if (s1.equals(s2)) {
            memo.put(s1 + "#" + s2, 1);
            return true;
        } 
        
        //num of letters
        int[] letter = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            letter[s1.charAt(i) - 'a']++;
            letter[s2.charAt(i) - 'a']--;
        }
        //diff num of letters -> false
        for (int i = 0; i < 26; i++) {
            if (letter[i] != 0) {
                memo.put(s1 + "#" + s2, 0);
                return false;
            }
        }
        
        //loop through all cut points
        for (int i = 1; i < s1.length(); i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                memo.put(s1 + "#" + s2, 1);
                return true;
            }
                
            //switch
            if (isScramble(s1.substring(i), s2.substring(0, s2.length() - i)) && isScramble(s1.substring(0, i), s2.substring(s2.length() - i))) {
                memo.put(s1 + "#" + s2, 1);
                return true;
            }
        }
        memo.put(s1 + "#" + s2, 0);
        return false;
    }
}
```

Solution 3 DP

- [solution detail](https://leetcode-cn.com/problems/scramble-string/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-1-2/)

```java
class Solution {
    public boolean isScramble(String s1, String s2) {
        //len of str
        if (s1.length() != s2.length()) return false;
        
        if (s1.equals(s2)) return true;
        
        //num of letters
        int[] letter = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            letter[s1.charAt(i) - 'a']++;
            letter[s2.charAt(i) - 'a']--;
        }
        //diff num of letters -> false
        for (int i = 0; i < 26; i++) {
            if (letter[i] != 0)
                return false;
        }
        
        int length = s1.length();
        boolean[][][] dp = new boolean[length + 1][length][length];
        //loop through all the len of str
        for (int len = 1; len < length + 1; len++) {
            //start of s1
            for (int i = 0; i + len < length + 1; i++) {
                //start of s2
                for (int j = 0; j + len < length + 1; j++) {
                    if (len == 1) {
                        dp[len][i][j] = s1.charAt(i) == s2.charAt(j);
                    } else {
                        //loop through all the cut point
                        for (int q = 1; q < len; q++) {
                            dp[len][i][j] = (dp[q][i][j] && dp[len-q][i+q][j+q]) || (dp[q][i][j+len-q] && dp[len-q][i+q][j]);
                            if (dp[len][i][j])
                                break;
                        }
                    }
                }
            }
        }
        return dp[length][0][0];
    }
}
```



### 125. Valid Palindrome (Easy) [@](https://leetcode.com/problems/valid-palindrome/submissions/)

> Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
>
> **Note:** For the purpose of this problem, we define empty string as valid palindrome.
>
> **Example 1:**
>
> ```
> Input: "A man, a plan, a canal: Panama"
> Output: true
> ```
>
> **Example 2:**
>
> ```
> Input: "race a car"
> Output: false
> ```

Solution 
```java
class Solution {
    public boolean isPalindrome(String s) {
        if (s.length() == 0 || s == null) return true;
        s = s.toLowerCase();
        int left = 0, right = s.length() - 1;
        while (left < right) {
            if (!Character.isLetterOrDigit(s.charAt(left))) {
                left ++;
                continue;
            }
            if (!Character.isLetterOrDigit(s.charAt(right))) {
                right --;
                continue;
            }
            if (s.charAt(left) == s.charAt(right)) {
                left ++;
                right --;
            }else {
                return false;
            }
        }
        return true;
    }
}
```



### 151. Reverse Words in a String (Medium)  [@](https://leetcode.com/problems/reverse-words-in-a-string/)

> Given an input string, reverse the string word by word.
>
> **Example 1:**
>
> ```
> Input: "the sky is blue"
> Output: "blue is sky the"
> ```
>
> **Example 2:**
>
> ```
> Input: "  hello world!  "
> Output: "world! hello"
> Explanation: Your reversed string should not contain leading or trailing spaces.
> ```
>
> **Example 3:**
>
> ```
> Input: "a good   example"
> Output: "example good a"
> Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
> ```

Solution 1 Library function

```java
class Solution {
    public String reverseWords(String s) {
        StringBuilder ans = new StringBuilder();
        //去掉s的首尾空格 然后将字符串拆分
        String[] str = s.trim().split(" ");
        for(int i = str.length - 1; i >= 0; i--){
            //空格后面的空格会变成空字符串
            if(!str[i].equals("")) ans.append(str[i] + " ");
        }
        //去掉最后添加上的空格
        ans = new StringBuilder(ans.toString().trim());
        return ans.toString();
    }
}
```

Solution 2 从后向前

- 将源字符串转换为数组，然后从后向前读取，遇到空格，判断是否单词结束

```JAVA
class Solution {
    public String reverseWords(String s) {
        if (s == null || s.length() == 0) return "";
        
        char[] chars = s.toCharArray();
        StringBuffer ans = new StringBuffer();
        int count = 0, ptr = s.length()-1;
        //从后向前扫描
        while(ptr >= 0) {
            //空格两种情况：单词结束 和 未开始
            if (chars[ptr] == ' ') {
                if (count != 0) {
                    ans.append(chars, ptr + 1, count).append(' ');
                    count = 0;//重置count
                }
            //遇到字母
            }else {
                count++;
            }
            ptr--;
        }
        //处理最后一个单词
        if (count != 0) {
            ans.append(chars, 0, count).append(' ');
        }
        //去除最后一个空格
        return ans.length() == 0 ? "" : ans.toString().substring(0, ans.length()-1);
    }
}
```

Solution 3

- 整句全部逆置，再逐个单词逆置

### 166. Fraction to Recurring Decimal (Medium) [@](https://leetcode.com/problems/fraction-to-recurring-decimal/)

> Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
>
> If the fractional part is repeating, enclose the repeating part in parentheses.
>
> **Example 1:**
>
> ```
> Input: numerator = 1, denominator = 2
> Output: "0.5"
> ```
>
> **Example 2:**
>
> ```
> Input: numerator = 2, denominator = 1
> Output: "2"
> ```
>
> **Example 3:**
>
> ```
> Input: numerator = 2, denominator = 3
> Output: "0.(6)"
> ```

Solution HashMap

- 利用哈希表存储余数位置，以判断是否存在重复的对应小数位
- 当出现重复则加入“（）”

```java
class Solution {
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        
        StringBuilder fraction = new StringBuilder();
        //positive or negative
        if (numerator < 0 ^ denominator < 0) {
            fraction.append("-");
        }
        
        Long dividend = Math.abs(Long.valueOf(numerator));
        Long divisor = Math.abs(Long.valueOf(denominator));
        //Integer Part
        fraction.append(String.valueOf(dividend/divisor));
        Long remainder = dividend % divisor;
        if (remainder == 0) {
            return fraction.toString();
        }
        //Decimal Part
        fraction.append(".");
        Map<Long, Integer> pos = new HashMap<>();
        while (remainder != 0) {
            //repeat
            if (pos.containsKey(remainder)) {
                fraction.insert(pos.get(remainder), "(");
                fraction.append(")");
                break;
            }
            //add new decimal num
            pos.put(remainder, fraction.length()); //add remainder's position
            remainder *= 10;
            fraction.append(String.valueOf(remainder / divisor));//add corresponding decimal
            remainder %= divisor;
        }
        return fraction.toString();
    }
}
```



### 168. Excel Sheet Column Title (Easy) [@](https://leetcode.com/problems/excel-sheet-column-title/)

> Given a positive integer, return its corresponding column title as appear in an Excel sheet.
>
> For example:
>
> ```
>  1 -> A
>  2 -> B
>  3 -> C
>  ...
>  26 -> Z
>  27 -> AA
>  28 -> AB 
>  ...
> ```
>
> **Example 1:**
>
> ```
> Input: 1
> Output: "A"
> ```
>
> **Example 2:**
>
> ```
> Input: 28
> Output: "AB"
> ```
>
> **Example 3:**
>
> ```
> Input: 701
> Output: "ZY"
> ```

Solution 进制转换

- 注意是1 - 26 所以使用 n-1

```java
class Solution {
    public String convertToTitle(int n) {
        StringBuilder str = new StringBuilder();
        while (n > 0) {
            str.insert(0, (char)((n-1)%26 + 'A'));
            n = (n-1)/26;
        }
        return str.toString();
    }
}
```



### 171. Excel Sheet Column Number (Easy) [@](https://leetcode.com/problems/excel-sheet-column-number/)

> Given a column title as appear in an Excel sheet, return its corresponding column number.
>
> For example:
>
> ```
>  A -> 1
>  B -> 2
>  C -> 3
>  ...
>  Z -> 26
>  AA -> 27
>  AB -> 28 
>  ...
> ```
>
> **Example 1:**
>
> ```
> Input: "A"
> Output: 1
> ```
>
> **Example 2:**
>
> ```
> Input: "AB"
> Output: 28
> ```
>
> **Example 3:**
>
> ```
> Input: "ZY"
> Output: 701
> ```

Solution

```java
class Solution {
    public int titleToNumber(String s) {
        int ans = 0;
        for (char c : s.toCharArray()) {
            ans = ans * 26 + (int)(c - 'A' + 1);
        }
        return ans;
    }
}
```



### 179. Largest Number (Medium) [@](https://leetcode.com/problems/largest-number/)

> Given a list of non negative integers, arrange them such that they form the largest number.
>
> **Example 1:**
>
> ```
> Input: [10,2]
> Output: "210"
> ```
>
> **Example 2:**
>
> ```
> Input: [3,30,34,5,9]
> Output: "9534330"
> ```
>
> **Note:** The result may be very large, so you need to return a string instead of an integer.

Solution 

```java
class Solution {
    public class LargerNumberComparator implements Comparator<String> {
        @Override
        public int compare(String a, String b) {
            String order1 = a + b;
            String order2 = b + a;
            return order2.compareTo(order1);
        }
    }
    
    public String largestNumber(int[] nums) {
        //convert to strings
        String[] asStrs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            asStrs[i] = String.valueOf(nums[i]);
        }
        
        Arrays.sort(asStrs, new LargerNumberComparator());
        
        if (asStrs[0].equals("0")) return "0";
        
        StringBuilder ans = new StringBuilder();
        for (String str : asStrs) {
            ans.append(str);
        }
        return ans.toString();
    }
}
```





### 205. Isomorphic Strings (Easy) [@](https://leetcode.com/problems/isomorphic-strings/)

> Given two strings ***s\*** and ***t\***, determine if they are isomorphic.
>
> Two strings are isomorphic if the characters in ***s\*** can be replaced to get ***t\***.
>
> All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.
>
> **Example 1:**
>
> ```
> Input: s = "egg", t = "add"
> Output: true
> ```
>
> **Example 2:**
>
> ```
> Input: s = "foo", t = "bar"
> Output: false
> ```
>
> **Example 3:**
>
> ```
> Input: s = "paper", t = "title"
> Output: true
> ```
>
> **Note:**
> You may assume both ***s\*** and ***t\*** have the same length.

Solution 1 HashMap

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) return false;
        //map -- match char in s and t
        HashMap<Character, Character> map = new HashMap<>();
        //set -- judge char in t whether has mapped
        Set<Character> set = new HashSet<>();
        char sChart, tChart;
        for (int i = 0; i < s.length(); i++) {
            sChart = s.charAt(i);
            tChart = t.charAt(i);
            
            if(!map.containsKey(sChart)) {//no mapped
                if (set.contains(tChart)) {//t has been mapped
                    return false;
                }else {
                    map.put(sChart, tChart);
                    set.add(tChart);
                }
            }else {//mapped
                if (map.get(sChart) != tChart) {
                    return false;
                }
            }
        }
        return true;
    }
}
```

Solution 2 Array

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        char[] sChars = s.toCharArray();
        char[] tChars = t.toCharArray();

        int length = sChars.length;
        if(length != tChars.length) return false;

        char[] sm = new char[256];
        char[] tm = new char[256];

        for(int i=0; i<length; i++){
            char sc = sChars[i];
            char tc = tChars[i];
            if(sm[sc] == 0 && tm[tc] == 0){
                sm[sc] = tc;
                tm[tc] = sc;
            }else{
                if(sm[sc] != tc || tm[tc] != sc){
                    return false;
                }
            }
        }
        return true;
    }
}
```



### 242. Valid Anagram (Easy) [@](https://leetcode.com/problems/valid-anagram/)

> Given two strings *s* and *t* , write a function to determine if *t* is an anagram of *s*.
>
> **Example 1:**
>
> ```
> Input: s = "anagram", t = "nagaram"
> Output: true
> ```
>
> **Example 2:**
>
> ```
> Input: s = "rat", t = "car"
> Output: false
> ```
>
> **Note:**
> You may assume the string contains only lowercase alphabets.
>
> **Follow up:**
> What if the inputs contain unicode characters? How would you adapt your solution to such case?

Solution 1 Sort

- 排序，比较

```java
class Solution {
    public boolean isAnagram(String s, String t) {
		char[] sArr = s.toCharArray();
		char[] tArr = t.toCharArray();
 
		Arrays.sort(sArr);
		Arrays.sort(tArr);
 
		return String.valueOf(sArr).equals(String.valueOf(tArr));
    }
}
```

Solution 2 Array

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        
        int[] count = new int[26];
        for (int i = 0; i < s.length(); i++) {
            char sChar = s.charAt(i);
            char tChar = t.charAt(i);
            
            count[sChar-'a'] ++;
            count[tChar-'a'] --;
        }
       	//遍历整个count数组
        for (int i = 0; i < count.length; i++) {
            if (count[i] != 0)
                return false;
        }
        return true;
    }
}
```



### 290. Word Pattern (Easy) [@](https://leetcode.com/problems/word-pattern/submissions/)

> Given a `pattern` and a string `str`, find if `str` follows the same pattern.
>
> Here **follow** means a full match, such that there is a bijection between a letter in `pattern` and a **non-empty** word in `str`.
>
> **Example 1:**
>
> ```
> Input: pattern = "abba", str = "dog cat cat dog"
> Output: true
> ```
>
> **Example 2:**
>
> ```
> Input:pattern = "abba", str = "dog cat cat fish"
> Output: false
> ```
>
> **Example 3:**
>
> ```
> Input: pattern = "aaaa", str = "dog cat cat dog"
> Output: false
> ```
>
> **Example 4:**
>
> ```
> Input: pattern = "abba", str = "dog dog dog dog"
> Output: false
> ```
>
> **Notes:**
> You may assume `pattern` contains only lowercase letters, and `str` contains lowercase letters that may be separated by a single space.

Solution--Improvement of P205 Solution 1

- 由于字符串比较需要转换，所以用字符串做键，模式字符作为值

```java
class Solution {
    public boolean wordPattern(String pattern, String str) {
        String[] Strs = str.split(" ");
        char[] p = pattern.toCharArray();
        if (Strs.length != p.length) return false;
        
        HashMap<String, Character> map = new HashMap<>();
        Set<Character> set = new HashSet<>();
        for (int i = 0; i < p.length; i++) {
            
            if(!map.containsKey(Strs[i])) {//no mapped
                if (set.contains(p[i])) {
                    return false;
                }else {
                    map.put(Strs[i], p[i]);
                    set.add(p[i]);
                }
            }else {//mapped
                if (map.get(Strs[i]) != p[i]) {
                    return false;
                }
            }
        }
        return true;
    }
}
```



### 316. Remove Duplicate Letters (Hard) [@](https://leetcode.com/problems/remove-duplicate-letters/submissions/)

> Given a string which contains only lowercase letters, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.
>
> **Example 1:**
>
> ```
> Input: "bcabc"
> Output: "abc"
> ```
>
> **Example 2:**
>
> ```
> Input: "cbacdcbc"
> Output: "acdb"
> ```

Solution 1 Stack

思路：每个字符必须出现一次，当这个字符只有一次机会的时候必须添加到字符串结尾，反之，如果后面还有则可以把优先级高的先放进来。
步骤：

1. 统计s中字符最后位置

2. 如果当前字符已经出现在stack中则跳过

3. 如果当前字符不在栈里：

   a. 若当前字符char小与栈顶元素，且栈顶元素有剩余 =》栈顶出栈 并 标记栈顶元素不在栈中（重复该操作直到不满足条件或栈为空）

   b. 当前字符char入栈，并标记char在栈中

```java
class Solution {
    public String removeDuplicateLetters(String s) {
        int len = s.length();
        if (len < 2) return s;
        
        boolean[] visited = new boolean[26];
        int[] lastPos = new int[26]; // 1
        
        for (int i = 0; i < len; i++) {
            lastPos[s.charAt(i) - 'a'] = i;
        }
        
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < len; i++) {
            char cur = s.charAt(i);
            // 2
            if (visited[cur - 'a']) continue;
            // 3.a
            while (!stack.isEmpty() && stack.peek() > cur && lastPos[stack.peek() - 'a'] > i) {
                visited[stack.pop() - 'a'] = false;
            }
            // 3.b
            stack.add(cur);
            visited[cur - 'a'] = true;
        }
        
        StringBuilder str = new StringBuilder();
        while(!stack.isEmpty()) {
            str.insert(0, stack.pop());
        }
        return str.toString();
    }
}
```

Solution 2 Array

- 思路和stack一样但是不需要用到stack

```java
class Solution {
    public String removeDuplicateLetters(String s) {
        int[] count = new int[26];
        boolean[] visited = new boolean[26];
        char[] charArr = s.toCharArray();
        
        for(char c : charArr) {
            count[c - 'a']++;
        }
        
        int i =0;
        for(char c : charArr) {
            count[c - 'a']--;
            if(visited[c- 'a']) continue;
            while(i >0 && charArr[i-1] >= c && count[charArr[i-1]-'a']>0){
                visited[charArr[i-1]-'a'] = false;
                i--;
            }
            charArr[i] = c;
            visited[c -'a'] = true;
            i++;
        }
        return new String(charArr).substring(0,i);
    }
}
```



### 344. Reverse String (Easy) [@](https://leetcode.com/problems/reverse-string/)

> Write a function that reverses a string. The input string is given as an array of characters `char[]`.
>
> Do not allocate extra space for another array, you must do this by **modifying the input array [in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** with O(1) extra memory.
>
> You may assume all the characters consist of [printable ascii characters](https://en.wikipedia.org/wiki/ASCII#Printable_characters).
>
> 
>
> **Example 1:**
>
> ```
> Input: ["h","e","l","l","o"]
> Output: ["o","l","l","e","h"]
> ```
>
> **Example 2:**
>
> ```
> Input: ["H","a","n","n","a","h"]
> Output: ["h","a","n","n","a","H"]
> ```

Solution Swap

```java
class Solution {
    public void reverseString(char[] s) {
        if (s.length == 0 || s == null) return;
        for (int i = 0; i < s.length/2; i++) {
            swap(s, i, s.length-i-1);
        }
    }
    private void swap(char[] s, int i, int j) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
```



### 345. Reverse Vowels of a String (Easy) [@](https://leetcode.com/problems/reverse-vowels-of-a-string/)

> Write a function that takes a string as input and reverse only the vowels of a string.
>
> **Example 1:**
>
> ```
> Input: "hello"
> Output: "holle"
> ```
>
> **Example 2:**
>
> ```
> Input: "leetcode"
> Output: "leotcede"
> ```
>
> **Note:**
> The vowels does not include the letter "y".

Solution 

```java
class Solution {
    public String reverseVowels(String s) {
        char[] arr = s.toCharArray();
        int left =0;
        int right =arr.length-1;
         
        while(left<right)
        {
            while(!isVowel(s.charAt(left))&&left<right)
                left++;
            while(!isVowel(s.charAt(right))&&left<right)
                right--;
            swap(left,right,arr);
            left++;
            right--;
        }
        return new String(arr);
         
    }
     
    public boolean isVowel(char c)
    {
        if(c=='a'||c=='e'||c=='i'||c=='o'||c=='u')
            return true;
        if(c=='A'||c=='E'||c=='I'||c=='O'||c=='U')
            return true;
        return false;
    }
     
    public void swap(int i,int j,char[] arr)
    {
        char tmp = arr[i];
        arr[i] =arr[j];
        arr[j]=tmp;
    }
}
```



### 383. Ransom Note (Easy) [@](https://leetcode.com/problems/ransom-note/)

> Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false.
>
> Each letter in the magazine string can only be used once in your ransom note.
>
> **Note:**
> You may assume that both strings contain only lowercase letters.
>
> ```
> canConstruct("a", "b") -> false
> canConstruct("aa", "ab") -> false
> canConstruct("aa", "aab") -> true
> ```

Solution 1 Hash Map

```java
class Solution {
    public boolean canConstruct(String ransomNote, String magazine) {
        //generate hash map of magazine
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < magazine.length(); i++) {
            char c = magazine.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        //update the num of characters in map
        for (int i = 0; i < ransomNote.length(); i++) {
            char c = ransomNote.charAt(i);
            if (!map.containsKey(c))
                return false;
            map.put(c, map.get(c)-1);
            if (map.get(c) < 0)
                return false;
        }
        return true;
    }
}
```

Solution 2 String to Array

- 字符转化为数字对应

```java
class Solution {
    public boolean canConstruct(String ransomNote, String magazine) {
        int count[] = new int[26];
        
        for(char ch: magazine.toCharArray()) {
            count[ch - 'a']++;
        }
        
        for(char ch: ransomNote.toCharArray()) {
            if(count[ch - 'a'] == 0) {
                return false;
            }
            count[ch - 'a']--;
        }
        return true;
    }
}
```

Solution 3

```java
class Solution {
    public boolean canConstruct(String ransomNote, String magazine) {
        //若magazine比ransomNote短则一定不可能
        if (magazine.length() < ransomNote.length()) return false;
        char[] r = ransomNote.toCharArray();

        int[] list = new int[26];
        int idx = 0;
        for (char c : r) {
            //判断mag中是否还有c
            //public int indexOf(int char, int fromIndex)
            idx = magazine.indexOf(c, list[c - 'a']);
            if (idx < 0) {
                return false;
            }
            //使用过之后向后移动idx
            list[c - 'a'] = idx + 1;
        }
        return true;
    }
}
```



### 387. First Unique Character in a String (Easy)  [@](https://leetcode.com/problems/first-unique-character-in-a-string/)

> Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
>
> **Examples:**
>
> ```
> s = "leetcode"
> return 0.
> 
> s = "loveleetcode",
> return 2.
> ```
>
> 
>
> **Note:** You may assume the string contain only lowercase letters.

Solution Hash Map

```java
class Solution {
    public int firstUniqChar(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        //generate and update hashmap
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        //traverse the map
        for (int i = 0; i < s.length(); i++) {
            if (map.get(s.charAt(i)) == 1)
                return i;
        }
        return -1;
    }
}
```



## Array

### 54. Spiral Matrix (Medium) [@](https://leetcode.com/problems/spiral-matrix/)

> Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.
>
> **Example 1:**
>
> ```
> Input:
> [
> [ 1, 2, 3 ],
> [ 4, 5, 6 ],
> [ 7, 8, 9 ]
> ]
> Output: [1,2,3,6,9,8,7,4,5]
> ```
>
> **Example 2:**
>
> ```
> Input:
> [
> [1, 2, 3, 4],
> [5, 6, 7, 8],
> [9,10,11,12]
> ]
> Output: [1,2,3,4,8,12,11,10,9,5,6,7]
> ```

Solution 1 Simulation
```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList();
        if (matrix == null || matrix.length == 0) return res;
        int R = matrix.length, C = matrix[0].length;
        boolean[][] visited = new boolean[R][C];
        int[] dr = {0, 1, 0, -1};
        int[] dc = {1, 0, -1, 0};
        int r = 0, c = 0, di = 0;
        for (int i = 0; i < R*C; i++) {
            res.add(matrix[r][c]);
            visited[r][c] = true;
            int cur_c = c + dc[di];
            int cur_r = r + dr[di];
            if (cur_c >= 0 && cur_c < C && cur_r >= 0 && cur_r < R && !visited[cur_r][cur_c]) {
                c = cur_c;
                r = cur_r;
            }else {
                di = (di + 1) % 4;
                c += dc[di];
                r += dr[di];
            }
        }
        return res;
    }
}
```

Solution 2 Layer by Layer
- For each outer layer, we want to iterate through its elements in clockwise order starting from the top left corner. Suppose the current outer layer has top-left coordinates (r1, c1) and bottom-right coordinates (r2, c2).
- top : c from c1 ...... c2
- right : r from r1+1 ...... r2
- bottom : c from c2-1 ...... c1+1
- left : r from r2 ...... r1-1
```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList();
        if (matrix == null || matrix.length == 0) return res;
        int r1 = 0, r2 = matrix.length - 1;
        int c1 = 0, c2 = matrix[0].length - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) res.add(matrix[r1][c]);
            for (int r = r1+1; r <= r2; r++) res.add(matrix[r][c2]);
            if (r1 < r2 && c1 < c2) {
                for (int c = c2 - 1; c > c1; c--) res.add(matrix[r2][c]);
                for (int r = r2; r > r1; r--) res.add(matrix[r][c1]);
            }
            //move top-left and bottom-right point
            r1++;
            c1++;
            r2--;
            c2--;
        }
        return res;
    }
}
```

### 56. Merge Intervals (Medium) [@](https://leetcode.com/problems/merge-intervals/)

> Given a collection of intervals, merge all overlapping intervals.
>
> **Example 1:**
>
> ```
> Input: [[1,3],[2,6],[8,10],[15,18]]
> Output: [[1,6],[8,10],[15,18]]
> Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
> ```
>
> **Example 2:**
>
> ```
> Input: [[1,4],[4,5]]
> Output: [[1,5]]
> Explanation: Intervals [1,4] and [4,5] are considered overlapping.
> ```
>
> **NOTE:** input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.

Solution Sort + Compare

- 按照start排序，然后相邻之间比较
- [Lambda表达式](https://blog.csdn.net/wqh8522/article/details/79745350)
```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length <= 1) return intervals;
        //sorted by start
        Arrays.sort(intervals,(i1, i2) -> Integer.compare(i1[0], i2[0]));
        
        List<int[]> res = new ArrayList<>();
        int[] newInterval = intervals[0];
        res.add(newInterval);
        for (int[] interval : intervals) {
            //前一个右界大于等于后一个左界
            if (newInterval[1] >= interval[0]) {
                newInterval[1] = Math.max(newInterval[1], interval[1]);
            }else {
                newInterval = interval;
                res.add(newInterval);
            }
        }
        return res.toArray(new int[res.size()][]);
    }
}
```

### 66. Plus One (Easy) [@](https://leetcode.com/problems/plus-one/)

> Given a **non-empty** array of digits representing a non-negative integer, plus one to the integer.
>
> The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
>
> You may assume the integer does not contain any leading zero, except the number 0 itself.
>
> **Example 1:**
>
> ```
> Input: [1,2,3]
> Output: [1,2,4]
> Explanation: The array represents the integer 123.
> ```
>
> **Example 2:**
>
> ```
> Input: [4,3,2,1]
> Output: [4,3,2,2]
> Explanation: The array represents the integer 4321.
> ```

Solution

```java
class Solution {
    public int[] plusOne(int[] digits) {
        for (int i = digits.length - 1; i >= 0; i--) {
            digits[i]++;
            digits[i] %= 10;
            //若进位则继续遍历，若不进位则直接返回
            if (digits[i] != 0) return digits;
        }
        //only 99,999...need one more digit
        digits = new int[digits.length+1];
        digits[0] = 1;
        return digits;
    }
}
```

### 73. Set Matrix Zeros (Medium) [@](https://leetcode.com/problems/set-matrix-zeroes/)

> Given a *m* x *n* matrix, if an element is 0, set its entire row and column to 0. Do it [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm).
>
> **Example 1:**
>
> ```
> Input: 
> [
> [1,1,1],
> [1,0,1],
> [1,1,1]
> ]
> Output: 
> [
> [1,0,1],
> [0,0,0],
> [1,0,1]
> ]
> ```
>
> **Example 2:**
>
> ```
> Input: 
> [
> [0,1,2,0],
> [3,4,5,2],
> [1,3,1,5]
> ]
> Output: 
> [
> [0,0,0,0],
> [0,4,5,0],
> [0,3,1,0]
> ]
> ```
>
> **Follow up:**
>
> - A straight forward solution using O(*m**n*) space is probably a bad idea.
> - A simple improvement uses O(*m* + *n*) space, but still not the best solution.
> - Could you devise a constant space solution?

Solution 
- Use first row and first column to mark if the row/col needs to be set to 0. (Postpone the change)
```java
class Solution {
  public void setZeroes(int[][] matrix) {
    Boolean isCol = false;
    int R = matrix.length;
    int C = matrix[0].length;

    for (int i = 0; i < R; i++) {

      // Since first cell for both first row and first column is the same i.e. matrix[0][0]
      // We can use an additional variable for either the first row/column.
      // For this solution we are using an additional variable for the first column
      // and using matrix[0][0] for the first row.
      if (matrix[i][0] == 0) {
        isCol = true;
      }

      for (int j = 1; j < C; j++) {
        // If an element is zero, we set the first element of the corresponding row and column to 0
        if (matrix[i][j] == 0) {
          matrix[0][j] = 0;
          matrix[i][0] = 0;
        }
      }
    }

    // Iterate over the array once again and using the first row and first column, update the elements.
    for (int i = 1; i < R; i++) {
      for (int j = 1; j < C; j++) {
        if (matrix[i][0] == 0 || matrix[0][j] == 0) {
          matrix[i][j] = 0;
        }
      }
    }

    // See if the first row needs to be set to zero as well
    if (matrix[0][0] == 0) {
      for (int j = 0; j < C; j++) {
        matrix[0][j] = 0;
      }
    }

    // See if the first column needs to be set to zero as well
    if (isCol) {
      for (int i = 0; i < R; i++) {
        matrix[i][0] = 0;
      }
    }
  }
}
```
### 75. Sort Colors (Medium) [@](https://leetcode.com/problems/sort-colors/)

> Given an array with *n* objects colored red, white or blue, sort them **[in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** so that objects of the same color are adjacent, with the colors in the order red, white and blue.
>
> Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
>
> **Note:** You are not suppose to use the library's sort function for this problem.
>
> **Example:**
>
> ```
> Input: [2,0,2,1,1,0]
> Output: [0,0,1,1,2,2]
> ```
>
> **Follow up:**
>
> - A rather straight forward solution is a two-pass algorithm using counting sort.
>   First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
> - Could you come up with a one-pass algorithm using only constant space?

Solution

- red replace forward, blue replace backward 
```java
class Solution {
    public void sortColors(int[] nums) {
        int red = 0, blue = nums.length - 1;
        int i = 0;
        while (red <= blue && i <= blue) {
            if (nums[i] == 0) {
                swap(nums, red, i);
                red++;
            }else if (nums[i] == 2) {
                swap(nums, i, blue);
                blue--;
                i--; //re-evaluate current element
            }
            i++;
        }
        return;
    }
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 88. Merge Sorted Array (Easy) [@](https://leetcode.com/problems/merge-sorted-array/)
> Given two sorted integer arrays *nums1* and *nums2*, merge *nums2* into *nums1* as one sorted array.
>
> **Note:**
>
> - The number of elements initialized in *nums1* and *nums2* are *m* and *n* respectively.
> - You may assume that *nums1* has enough space (size that is greater or equal to *m* + *n*) to hold additional elements from *nums2*.
>
> **Example:**
>
> ```
> Input:
> nums1 = [1,2,3,0,0,0], m = 3
> nums2 = [2,5,6],       n = 3
> 
> Output: [1,2,2,3,5,6]
> ```

Solution

- 从后向前扫描，添加大的元素
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        //two pointer for nums1 and nums2
        int ptr1 = m - 1;
        int ptr2 = n - 1;
        //pointer for insert position
        int p = m + n - 1;
        //compare back forward
        while (ptr1 >= 0 && ptr2 >= 0) {
            if (nums1[ptr1] > nums2[ptr2]) {
                nums1[p--] = nums1[ptr1--];
            }else {
                nums1[p--] = nums2[ptr2--];
            }
        }
        //add remaining nums in nums2
        //public static void arraycopy(Object src, int srcPos, Object dest, int destPos, int length)
        System.arraycopy(nums2, 0, nums1, 0, ptr2 + 1);
    }
}
```

### 118. Pascal‘s Triangle (Easy) [@](https://leetcode.com/problems/pascals-triangle/)

> Given a non-negative integer *numRows*, generate the first *numRows* of Pascal's triangle.
>
> In Pascal's triangle, each number is the sum of the two numbers directly above it.
>
> **Example:**
>
> ```
> Input: 5
> Output:
> [
>   [1],
>  [1,1],
> [1,2,1],
> [1,3,3,1],
> [1,4,6,4,1]
> ]
> ```

Solution DP

```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();

        // First base case; if user requests zero rows, they get zero rows.
        if (numRows == 0) {
            return triangle;
        }

        // Second base case; first row is always [1].
        triangle.add(new ArrayList<>());
        triangle.get(0).add(1);

        for (int rowNum = 1; rowNum < numRows; rowNum++) {
            List<Integer> row = new ArrayList<>(Integer);
            List<Integer> prevRow = triangle.get(rowNum-1);

            // The first row element is always 1.
            row.add(1);

            // Each triangle element (other than the first and last of each row)
            // is equal to the sum of the elements above-and-to-the-left and
            // above-and-to-the-right.
            for (int j = 1; j < rowNum; j++) {
                row.add(prevRow.get(j-1) + prevRow.get(j));
            }

            // The last row element is always 1.
            row.add(1);

            triangle.add(row);
        }

        return triangle;
    }
}
```

### 136. Single Number (Easy) [@](https://leetcode.com/problems/single-number/)

> Given a **non-empty** array of integers, every element appears *twice* except for one. Find that single one.
>
> **Note:**
>
> Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
>
> **Example 1:**
>
> ```
> Input: [2,2,1]
> Output: 1
> ```
>
> **Example 2:**
>
> ```
> Input: [4,1,2,1,2]
> Output: 4
> ```

Solution 1 Hash Map

```java
class Solution {
    public int singleNumber(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                map.remove(num);
            }else {
                map.put(num, 1);
            }
        }
        return map.entrySet().iterator().next().getKey();
    }
}
```

Solution 2 Math
- 所有不重复数字的两倍乘总和 - 原数组总和 = 出现一次的数字
```java
class Solution {
    public int singleNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        
        int arrSum = 0;
        for (int num : nums) {
            set.add(num);
            arrSum += num;
        }
        
        int doubleSum = 0;
        for (int num : set) {
            doubleSum += 2*num;
        }
        return doubleSum - arrSum;
    }
}
```

Solution 3 Bit Manipulation
- 异或XOR
- 对所有数字进行异或，最后得出单个的数
- 异或性质
	- 交换律: A XOR B = B XOR A
	- 结合律: A XOR B XOR C = A XOR (B XOR C) = (A XOR B) XOR C
	- 自反性: A XOR B XOR B = A XOR 0 = A
```java
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;
    }
}
```

### 169. Majority Element (Easy) [@](https://leetcode.com/problems/majority-element/)

> Given an array of size *n*, find the majority element. The majority element is the element that appears **more than** `⌊ n/2 ⌋` times.
>
> You may assume that the array is non-empty and the majority element always exist in the array.
>
> **Example 1:**
>
> ```
> Input: [3,2,3]
> Output: 3
> ```
>
> **Example 2:**
>
> ```
> Input: [2,2,1,1,1,2,2]
> Output: 2
> ```

Solution 1 HashMap

```java
class Solution {
    public int majorityElement(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int N = nums.length;
        for (int i = 0; i < N; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        int res = 0;
        for (Map.Entry<Integer,Integer> s : map.entrySet()) {
            if (s.getValue() > N/2) {
                res = s.getKey();
                break;
            }
        }
        return res;
    }
}
```

Solution 2 Sort

```java
class Solution {
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length/2];
    }
}
```
Solution 3 摩尔投票法

- 首先假定数组头元素即为众数，设定计数器为1，从第二个数开始遍历，如果和头元素相同则计数器加1，如果不相同则减1，之后判断计数器是否为0，如果不为0则继续下一步循环，如果为0则将众数指针指向当前元素，以此类推，最后众数指针指向的元素即为众数，时间复杂度～O(n)，空间复杂度～O(1)

```java
class Solution {
    public int majorityElement(int[] nums) {
        int res = nums[0];
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == res) count++;
            else {
                count--;
                if (count == 0) {
                    res = nums[i];
                    count = 1;
                }
            }
        }
        return res;
    }
}
```
### 189. Rotate Array (Easy) [@](https://leetcode.com/problems/rotate-array/)

> Given an array, rotate the array to the right by *k* steps, where *k* is non-negative.
>
> **Example 1:**
>
> ```
> Input: [1,2,3,4,5,6,7] and k = 3
> Output: [5,6,7,1,2,3,4]
> Explanation:
> rotate 1 steps to the right: [7,1,2,3,4,5,6]
> rotate 2 steps to the right: [6,7,1,2,3,4,5]
> rotate 3 steps to the right: [5,6,7,1,2,3,4]
> ```
>
> **Example 2:**
>
> ```
> Input: [-1,-100,3,99] and k = 2
> Output: [3,99,-1,-100]
> Explanation: 
> rotate 1 steps to the right: [99,-1,-100,3]
> rotate 2 steps to the right: [3,99,-1,-100]
> ```
>
> **Note:**
>
> - Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
> - Could you do it in-place with O(1) extra space?

Solution 1 Extra Array

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int N = nums.length;
        int[] temp = new int[N];
        for (int i = 0; i < N; i++) {
            temp[(i+k) % N] = nums[i];
        }
        for (int i = 0; i < N; i++) {
            nums[i] = temp[i];
        }
    }
}
```

Solution 2 Cyclic Replacements

- [Leetcode CN](https://leetcode-cn.com/problems/rotate-array/solution/xuan-zhuan-shu-zu-by-leetcode/)

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int count = 0;
        for (int start = 0; count < nums.length; start++) {
            int cur = start;
            int prev = nums[cur];
            do {
                int next = (cur + k) % nums.length;
                int temp = nums[next];
                nums[next] = prev;
                prev = temp;
                cur = next;
                count++;
            }while (start != cur);
        }
    }
}
```
Solution 3 Reverse
- 先反转所有，再分别反转前k个位置和后面所有
```java
class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }
}
```


## Linked List

### 138. Copy List with Random Pointer (Medium) [@](https://leetcode.com/problems/copy-list-with-random-pointer/)

> A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
>
> Return a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) of the list.
>
> The Linked List is represented in the input/output as a list of `n` nodes. Each node is represented as a pair of `[val, random_index]` where:
>
> - `val`: an integer representing `Node.val`
> - `random_index`: the index of the node (range from `0` to `n-1`) where random pointer points to, or `null` if it does not point to any node.
>
> **Example 1:**
>
> ![img](https://assets.leetcode.com/uploads/2019/12/18/e1.png)
>
> ```
> Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
> Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
> ```
>
> **Example 2:**
>
> ![img](https://assets.leetcode.com/uploads/2019/12/18/e2.png)
>
> ```
> Input: head = [[1,1],[2,1]]
> Output: [[1,1],[2,1]]
> ```
>
> **Example 3:**
>
> **![img](https://assets.leetcode.com/uploads/2019/12/18/e3.png)**
>
> ```
> Input: head = [[3,null],[3,0],[3,null]]
> Output: [[3,null],[3,0],[3,null]]
> ```
>
> **Example 4:**
>
> ```
> Input: head = []
> Output: []
> Explanation: Given linked list is empty (null pointer), so return null.
> ```

Solution 1 HashMap + 2 iterations 

第一种方法，就是使用HashMap来坐，HashMap的key存原始pointer，value存新的pointer。

- 第一遍，先不copy random的值，只copy数值建立好新的链表。并把新旧pointer存在HashMap中。
- 第二遍，遍历旧表，复制random的值，因为第一遍已经把链表复制好了并且也存在HashMap里了，所以只需从HashMap中，把当前旧的node.random作为key值，得到新的value的值，并把其赋给新node.random就好。

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        HashMap<Node, Node> map = new HashMap<>();
        Node newHead = new Node(head.val);
        map.put(head, newHead);
        Node oldPtr = head.next;
        Node newPtr = newHead;
        //first iterate the linked list
        while (oldPtr != null) {
            //link next node
            Node newNode = new Node(oldPtr.val);
            newPtr.next = newNode;
            //update hashmap
            map.put(oldPtr, newNode);
            
            oldPtr = oldPtr.next;
            newPtr = newPtr.next;
        }
        
        oldPtr = head;
        newPtr = newHead;
        //second iterate the linked list -> update random ptr
        while(oldPtr != null) {
            //update random ptr of new list
            newPtr.random = map.get(oldPtr.random);
            
            oldPtr = oldPtr.next;
            newPtr = newPtr.next;
        }
        
        return newHead;
    }
}
```
Solution 2 3-iteration

第二种方法不使用HashMap来做，使空间复杂度降为O(1)，不过需要3次遍历list，时间复杂度为O(3n)=O(n)。

- 第一遍，对每个node进行复制，并插入其原始node的后面，新旧交替，变成重复链表。如：原始：1->2->3->null，复制后：1->1->2->2->3->3->null
- 第二遍，遍历每个旧node，把旧node的random的复制给新node的random，因为链表已经是新旧交替的。所以复制方法为：**node.next.random = node.random.next** 前面是说旧node的next的random，就是新node的random，后面是旧node的random的next，正好是新node，是从旧random复制来的。
- 第三遍，则是把新旧两个表拆开，返回新的表即可。

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Node cur = head;
        //1st iteration: copy current node and link it to next
        while (cur != null) {
            Node newNode = new Node(cur.val);
            newNode.next = cur.next;
            cur.next = newNode;
            
            cur = newNode.next;
        }
        //2ed iteration: update random node
        cur = head;
        while (cur != null) {
            if (cur.random != null) {
                cur.next.random = cur.random.next;
            }
            cur = cur.next.next;
        }
        //3rd iteration: 
        cur = head; //point to old node
        Node newHead = head.next;//initialize new head
        Node copy = newHead;//point to new node
        
        // method 1
//         while (copy.next != null) {
//             //old
//             cur.next = cur.next.next;
//             cur = cur.next;
            
//             //new
//             copy.next = copy.next.next;
//             copy = copy.next;
//         }
//         cur.next = cur.next.next;
        
        //method 2
        while (cur != null) {
            copy = cur.next;
            cur.next = copy.next;
            if (copy.next != null) {
                copy.next = copy.next.next;
            }
            cur = cur.next;
        }
        return newHead;
    }
}
```

### 141. Linked List Cycle (Easy) [@](https://leetcode.com/problems/linked-list-cycle/)

> Given a linked list, determine if it has a cycle in it.
>
> To represent a cycle in the given linked list, we use an integer `pos` which represents the position (0-indexed) in the linked list where tail connects to. If `pos` is `-1`, then there is no cycle in the linked list.
>
> 
>
> **Example 1:**
>
> ```
> Input: head = [3,2,0,-4], pos = 1
> Output: true
> Explanation: There is a cycle in the linked list, where tail connects to the second node.
> ```
>
> ![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)
>
> **Example 2:**
>
> ```
> Input: head = [1,2], pos = 0
> Output: true
> Explanation: There is a cycle in the linked list, where tail connects to the first node.
> ```
>
> ![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)
>
> **Example 3:**
>
> ```
> Input: head = [1], pos = -1
> Output: false
> Explanation: There is no cycle in the linked list.
> ```
>
> ![img](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test3.png)
>
> 
>
> **Follow up:**
>
> Can you solve it using *O(1)* (i.e. constant) memory?

Solution 1 HashSet

```java
public boolean hasCycle(ListNode head) {
    Set<ListNode> nodesSeen = new HashSet<>();
    while (head != null) {
        if (nodesSeen.contains(head)) {
            return true;
        } else {
            nodesSeen.add(head);
        }
        head = head.next;
    }
    return false;
}
```

Solution 2 Two ptrs
- if slow and fast ptr meet then it much contains a cycle.
```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null)
                return false;
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }
}
```
### 146. LRU Cache (Medium) [@](https://leetcode.com/problems/lru-cache/)

> Design and implement a data structure for [Least Recently Used (LRU) cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU). It should support the following operations: `get` and `put`.
>
> `get(key)` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
> `put(key, value)` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.
>
> The cache is initialized with a **positive** capacity.
>
> **Follow up:**
> Could you do both operations in **O(1)** time complexity?
>
> **Example:**
>
> ```
> LRUCache cache = new LRUCache( 2 /* capacity */ );
> 
> cache.put(1, 1);
> cache.put(2, 2);
> cache.get(1);       // returns 1
> cache.put(3, 3);    // evicts key 2
> cache.get(2);       // returns -1 (not found)
> cache.put(4, 4);    // evicts key 1
> cache.get(1);       // returns -1 (not found)
> cache.get(3);       // returns 3
> cache.get(4);       // returns 4
> ```

Solution 1 LinkedHashMap

- [LinkedHashMap](https://docs.oracle.com/javase/8/docs/api/java/util/LinkedHashMap.html)
- LinkedHashMap(int initialCapacity, float loadFactor, boolean accessOrder)
- 作为一般规则，默认负载因子（0.75）在时间和空间成本上提供了很好的折衷。较高的值会降低空间开销，但提高查找成本（体现在大多数的HashMap类的操作，包括get和put）。设置初始大小时，应该考虑预计的entry数在map及其负载系数，并且尽量减少rehash操作的次数。如果初始容量大于最大条目数除以负载因子，rehash操作将不会发生。


```java
class LRUCache extends LinkedHashMap<Integer, Integer>{
    private int capacity;
    
    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }
    
    public int get(int key) {
        return super.getOrDefault(key, -1);
    }
    
    public void put(int key, int value) {
        super.put(key, value);
    }
    
    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```
Solution 2 Linked List + HashMap (implementation of sol 1)

```java

class LRUCache {
    
    class DLinkedNode {
        int key;
        int val;
        DLinkedNode prev;
        DLinkedNode next;
    }
    
    private void addNode(DLinkedNode node) {
        //always add node right after head
        node.next = head.next;
        node.prev = head;
        
        head.next.prev = node;
        head.next = node;
    }
    
    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addNode(node);
    }
    
    private DLinkedNode popTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
    
    private HashMap<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;
    
    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        
        head = new DLinkedNode();
        tail = new DLinkedNode();
        
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        DLinkedNode node = cache.get(key);
        
        if (node == null) 
            return -1;
        else {
            //used => move to head
            moveToHead(node);
            return node.val;
        }
    }
    
    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        
        if (node == null) {
            DLinkedNode newNode = new DLinkedNode();
            newNode.key = key;
            newNode.val = value;
            
            cache.put(key, newNode);//update cache(HashMap)
            addNode(newNode); //update Linked List
            
            size++;
            
            if (size > capacity) {
                //pop tail
                DLinkedNode tail = popTail();
                cache.remove(tail.key);
                size--;
            }
        }else {
            //update val
            node.val = value;
            moveToHead(node);
        }
    }
    
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```

### 148. Sort List (Medium) [@](https://leetcode.com/problems/sort-list/)

> Sort a linked list in *O*(*n* log *n*) time using constant space complexity.
>
> **Example 1:**
>
> ```
> Input: 4->2->1->3
> Output: 1->2->3->4
> ```
>
> **Example 2:**
>
> ```
> Input: -1->5->3->4->0
> Output: -1->0->3->4->5
> ```

Solution 1 Merge sort
- 空间复杂度O(logn)， 不符合要求
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        
        //find the mid point
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //mid = slow
        ListNode tmp = slow.next;
        slow.next = null; //cut
        //divide
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);
        //merge
        ListNode cur = new ListNode(0);
        ListNode res = cur;
        while (left != null && right != null) {
            if (left.val < right.val) {
                cur.next = left;
                left = left.next;
            }else {
                cur.next = right;
                right = right.next;
            }
            cur = cur.next;
        }
        //add the rest of left or right part
        cur.next = left != null ? left : right;
        return res.next;
    }
}
```
Solution 2 Iteration
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        //len of list
        int length = 0;
        while (head != null) {
            length++;
            head = head.next;
        }
        head = dummy.next;
        
        //loop logn times
        for (int i = 1; i < length; i+= i) {
            //list was divided into 4 parts:
            //1. already sorted; 2. left part of list to be sorted;
            //3. right part of list to be sorted; 4. unsorted part of list
            ListNode success = dummy;
            ListNode left = null;
            ListNode right = null;
            while (head != null) {
                left = head;
                head = cutFromHead(head, i);
                right = head;
                head = cutFromHead(head, i);
                //merge sort left and right, put them after success, and update success
                success.next = mergeLists(left, right);
                while (success.next != null) success = success.next;
            }
            head = dummy.next;
        }
        return dummy.next;
    }
    //cut the list from head with n-len and return (n+1)th node
    private ListNode cutFromHead(ListNode head, int n) {
        while (head != null && --n > 0) {
            head = head.next;
        }
        
        if (head == null) return null;
        
        ListNode next = head.next;
        head.next = null; //cut
        return next;
    }
    //merge sort 2 lists
    private ListNode mergeLists(ListNode left, ListNode right) {
        if (left == null) return right;
        if (right == null) return left;
        
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        
        while (left != null && right != null) {
            if (left.val < right.val) {
                cur.next = left;
                left = left.next;
            }else {
                cur.next = right;
                right = right.next;
            }
            cur = cur.next;
        }
        cur.next = left != null ? left : right;
        return dummy.next;
    }
}
```
### 160. Intersection of Two Linked Lists (Easy) [@](https://leetcode.com/problems/intersection-of-two-linked-lists/)

> Write a program to find the node at which the intersection of two singly linked lists begins.
>
> For example, the following two linked lists:
>
> [![img](https://assets.leetcode.com/uploads/2018/12/13/160_statement.png)](https://assets.leetcode.com/uploads/2018/12/13/160_statement.png)
>
> begin to intersect at node c1.
>
> 
>
> **Example 1:**
>
> [![img](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)
>
> ```
> Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
> Output: Reference of the node with value = 8
> Input Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,0,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
> ```
>
> 
>
> **Example 2:**
>
> [![img](https://assets.leetcode.com/uploads/2018/12/13/160_example_2.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_2.png)
>
> ```
> Input: intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
> Output: Reference of the node with value = 2
> Input Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [0,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
> ```
>
> 
>
> **Example 3:**
>
> [![img](https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png)](https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png)
>
> ```
> Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
> Output: null
> Input Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
> Explanation: The two lists do not intersect, so return null.
> ```
>
> 
>
> **Notes:**
>
> - If the two linked lists have no intersection at all, return `null`.
> - The linked lists must retain their original structure after the function returns.
> - You may assume there are no cycles anywhere in the entire linked structure.
> - Your code should preferably run in O(n) time and use only O(1) memory.

Solution 1 Two round iteration
- 尾部对齐，找出长度差，长的先走
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenA=0, lenB=0;
        ListNode A=headA, B=headB;
        while(A != null){
            lenA ++;
            A = A.next;
        }

        while(B != null) {
            lenB++;
            B = B.next;
        }
        A = headA;
        B = headB;

        while(lenA > lenB){
            A = A.next;
            lenA --;
        }
        while(lenA < lenB){
            B = B.next;
            lenB --;
        }

        while(A != B){
            A = A.next;
            B = B.next;
        }

        return A;
    }
}
```

Solution 2 Two pointer + 环
- 转化成判断环的问题
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        
        ListNode curA = headA;
        ListNode curB = headB;
        //将自身最后连接到另一个链表头，若存在intersection则形成环
        //如果形成环，则curA和curB一定在环的入口相交
        //若没有形成环，则都到达null
        while (curA != curB) {
            curA = curA == null ? headB : curA.next;
            curB = curB == null ? headA : curB.next;
        }
        return curA;
    }
}
```

## Stack

### 150. Evaluate Reverse Polish Notation (Medium) [@](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

> Evaluate the value of an arithmetic expression in [Reverse Polish Notation](http://en.wikipedia.org/wiki/Reverse_Polish_notation).
>
> Valid operators are `+`, `-`, `*`, `/`. Each operand may be an integer or another expression.
>
> **Note:**
>
> - Division between two integers should truncate toward zero.
> - The given RPN expression is always valid. That means the expression would always evaluate to a result and there won't be any divide by zero operation.
>
> **Example 1:**
>
> ```
> Input: ["2", "1", "+", "3", "*"]
> Output: 9
> Explanation: ((2 + 1) * 3) = 9
> ```
>
> **Example 2:**
>
> ```
> Input: ["4", "13", "5", "/", "+"]
> Output: 6
> Explanation: (4 + (13 / 5)) = 6
> ```
>
> **Example 3:**
>
> ```
> Input: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
> Output: 22
> Explanation: 
>   ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
> = ((10 * (6 / (12 * -11))) + 17) + 5
> = ((10 * (6 / -132)) + 17) + 5
> = ((10 * 0) + 17) + 5
> = (0 + 17) + 5
> = 17 + 5
> = 22
> ```

Solution 1 Stack
```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String s : tokens) {
            if (s.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (s.equals("-")) {
                stack.push(- stack.pop() + stack.pop());
            } else if (s.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (s.equals("/")) {
                int dividend = stack.pop();
                stack.push(stack.pop() / dividend);
            } else { //normal numbers
                stack.push(Integer.parseInt(s));
            }
        }
        return stack.pop();
    }
}
```

### 155. Min Stack (Medium) [@](https://leetcode.com/problems/min-stack/submissions/)

> Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
>
> - push(x) -- Push element x onto stack.
> - pop() -- Removes the element on top of the stack.
> - top() -- Get the top element.
> - getMin() -- Retrieve the minimum element in the stack.
>
>  
>
> **Example:**
>
> ```
> MinStack minStack = new MinStack();
> minStack.push(-2);
> minStack.push(0);
> minStack.push(-3);
> minStack.getMin();   --> Returns -3.
> minStack.pop();
> minStack.top();      --> Returns 0.
> minStack.getMin();   --> Returns -2.
> ```

Solution 1 synchronous data stack and helper stack
```java
class MinStack {
    
    private Stack<Integer> data;
    private Stack<Integer> minElement;

    /** initialize your data structure here. */
    public MinStack() {
        data = new Stack<>();
        minElement = new Stack<>();
    }
    
    public void push(int x) {
        data.add(x);
        if (minElement.isEmpty() || minElement.peek() >= x) {
            minElement.add(x);
        } else {
            minElement.add(minElement.peek());
        }
    }
    
    public void pop() {
        if (!data.isEmpty()) {
            data.pop();
            minElement.pop();
        }
    }
    
    public int top() {
        if (!data.isEmpty()) {
            return data.peek();
        }
        throw new RuntimeException("stack is empty");
    }
    
    public int getMin() {
        if (!minElement.isEmpty()) {
            return minElement.peek();
        }
        throw new RuntimeException("stack is empty");
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

Solution 2 asynchronous 

```java
class MinStack {
    
    private Stack<Integer> data;
    private Stack<Integer> minElement;

    /** initialize your data structure here. */
    public MinStack() {
        data = new Stack<>();
        minElement = new Stack<>();
    }
    
    // 思路 2：辅助栈和数据栈不同步
    // 关键 1：辅助栈的元素空的时候，必须放入新进来的数
    // 关键 2：新来的数小于或者等于辅助栈栈顶元素的时候，才放入（特别注意这里等于要考虑进去）
    // 关键 3：出栈的时候，辅助栈的栈顶元素等于数据栈的栈顶元素，才出栈，即"出栈保持同步"就可以了
    
    public void push(int x) {
        data.add(x);
        // 辅助栈在必要的时候才增加
        if (minElement.isEmpty() || minElement.peek() >= x) {
            minElement.add(x);
        }
    }
    
    public void pop() {
        //关键3: 数据栈一定pop
        if (!data.isEmpty()) {
            // 注意：声明成 int 类型，这里完成了自动拆箱，从 Integer 转成了 int，因此下面的比较可以使用 "==" 运算符
            // 参考资料：https://www.cnblogs.com/GuoYaxiang/p/6931264.html
            // 如果把 top 变量声明成 Integer 类型，下面的比较就得使用 equals 方法
            int top = data.pop();
            if (top == minElement.peek()) {
                minElement.pop();
            }
        }
    }
    
    public int top() {
        if (!data.isEmpty()) {
            return data.peek();
        }
        throw new RuntimeException("stack is empty");
    }
    
    public int getMin() {
        if (!minElement.isEmpty()) {
            return minElement.peek();
        }
        throw new RuntimeException("stack is empty");
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```
Solution 3 Linked List

```java
class MinStack {
    
    private Node head;
    
    private class Node {
        int val;
        int min;
        Node next;
        
        Node(int val, int min, Node next) {
            this.val = val;
            this.min = min;
            this.next = next;
        }
    }
    
    public void push(int x) {
        if (head == null) {
            head = new Node(x, x, null);
        } else {
            head = new Node(x, Math.min(head.min, x), head);
        }
    }
    
    public void pop() {
        head = head.next;
    }
    
    public int top() {
        return head.val;
    }
    
    public int getMin() {
        return head.min;
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

## Math

### 172. Factorial Trailing Zeroes (Easy) [@](https://leetcode.com/problems/factorial-trailing-zeroes/)

> Given an integer *n*, return the number of trailing zeroes in *n*!.
>
> **Example 1:**
>
> ```
> Input: 3
> Output: 0
> Explanation: 3! = 6, no trailing zero.
> ```
>
> **Example 2:**
>
> ```
> Input: 5
> Output: 1
> Explanation: 5! = 120, one trailing zero.
> ```
>
> **Note:** Your solution should be in logarithmic time complexity.

Solution 

- 判断n!中有多少个5
- [Detail](https://leetcode-cn.com/problems/factorial-trailing-zeroes/solution/xiang-xi-tong-su-de-si-lu-fen-xi-by-windliang-3/)

```java
public int trailingZeroes(int n) {
    int count = 0;
    while (n > 0) {
        count += n / 5;
        n = n / 5;
    }
    return count;
}
```
### 190. Reverse Bits (Easy) [@](https://leetcode.com/problems/reverse-bits/)

> Reverse bits of a given 32 bits unsigned integer.
>
> 
>
> **Example 1:**
>
> ```
> Input: 00000010100101000001111010011100
> Output: 00111001011110000010100101000000
> Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
> ```
>
> **Example 2:**
>
> ```
> Input: 11111111111111111111111111111101
> Output: 10111111111111111111111111111111
> Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.
> ```
>
> 
>
> **Note:**
>
> - Note that in some languages such as Java, there is no unsigned integer type. In this case, both input and output will be given as signed integer type and should not affect your implementation, as the internal binary representation of the integer is the same whether it is signed or unsigned.
> - In Java, the compiler represents the signed integers using [2's complement notation](https://en.wikipedia.org/wiki/Two's_complement). Therefore, in **Example 2** above the input represents the signed integer `-3` and the output represents the signed integer `-1073741825`.
>
>  
>
> **Follow up**:
>
> If this function is called many times, how would you optimize it?

Solution 1 shift

```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int res = 0;
        int count = 0;
        while (count < 32) {
            res <<= 1; //res is shifted by 1 position to left
            res |= (n & 1); //just pick the last pos of n to OR res
            n >>= 1; //n is shifted by 1 pos to right => get rid of the last pos
            count++;
        }
        return res;
    }
}
```
### 191. Number of 1 Bits (Easy) [@](https://leetcode.com/problems/number-of-1-bits/)

> Write a function that takes an unsigned integer and return the number of '1' bits it has (also known as the [Hamming weight](http://en.wikipedia.org/wiki/Hamming_weight)).
>
> 
>
> **Example 1:**
>
> ```
> Input: 00000000000000000000000000001011
> Output: 3
> Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
> ```
>
> **Example 2:**
>
> ```
> Input: 00000000000000000000000010000000
> Output: 1
> Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.
> ```
>
> **Example 3:**
>
> ```
> Input: 11111111111111111111111111111101
> Output: 31
> Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.
> ```
>
> 
>
> **Note:**
>
> - Note that in some languages such as Java, there is no unsigned integer type. In this case, the input will be given as signed integer type and should not affect your implementation, as the internal binary representation of the integer is the same whether it is signed or unsigned.
> - In Java, the compiler represents the signed integers using [2's complement notation](https://en.wikipedia.org/wiki/Two's_complement). Therefore, in **Example 3** above the input represents the signed integer `-3`.
>
>  
>
> **Follow up**:
>
> If this function is called many times, how would you optimize it?

Solution 1

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & 1) != 0) {
                res++;
            }
            n >>= 1;
        }
        return res;
    }
}
```
Solution 2 Flip
```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        while (n != 0) {
            res++;
            //always flip the least-significant 1-bit to 0
            n &= (n - 1);
        }
        return res;
    }
}
```

## Graph

## Sliding Window

## Partition