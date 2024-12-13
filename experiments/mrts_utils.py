import torch
''' compact_observation_space takes in a tensor of batch_size, height * width, 29 and returns the
same with final dim reduced to 6'''

def compact_observation_space(x):
	split_dims = [5,5,3,8,6,2]
	split_dims = torch.split(x, split_dims, dim=-1)
	new = []
	for dim in split_dims:
		new.append(torch.argmax(dim, dim=-1))
	return torch.cat(new, dim=-1)

def main():
	x = torch.randn((1, 16, 29))
	print(compact_observation_space(x).shape)
	print(compact_observation_space(x))
	# Expected output: torch.Size([1, 16, 6])

if __name__ == "__main__":
	main()