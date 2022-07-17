import torch

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                         dtype=torch.float32,
                         device='cpu')

print(my_tensor.shape)

x = torch.empty(size=(3, 3))  # 3x3 matrix
y = torch.rand((3, 3))
z = torch.eye(5,5)
xx = torch.linspace(start=.1, end=1, steps=20) #steps = how many numbers between start and end

print(xx)

batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand(batch, n, m)
t2 = torch.rand(batch, m, p)

out_batch_mult = torch.bmm(t1, t2)

x = torch.rand(10,25)
print(x)
print("//////////////////////////////////////////")
print(x[2][0:10])


