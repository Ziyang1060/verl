import ray
ray.init()

d = ray.state.available_resources_per_node()
print(len(d))