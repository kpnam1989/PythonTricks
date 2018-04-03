I usually use Spyder to run and experiment with my codes, and only transfer them to a Python Notebook later. The reason is that the design of the notebook makes it difficult to switch between running codes and editing. In Spyder, it is easy to do this with a shortcut. Today, I discovered that there is a way to make it much easier: connect to Kernel from Notebook to Spyder. 

First, we need to identify the Kernel name that is being run in your Jupyeter Notebook. In a new cell in the Notebook, run:
```python
%connect_info
```
and then you can see all the instruction. Alternatively, you can open Anaconda prompt and run:
```python
jupyter console --existing
```
this will help you connect directly to the kernel from Jupyter Console.

Once we know the Kernel name, we can go back to Spyder, open Consoles >> Connect to existing kernel and select the right files. Now the kernel between your Notebook and Spyder.
