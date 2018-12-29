from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import setting
 
linear = {('x0', 'x0'): -1, ('x1', 'x1'): -1, ('x2', 'x2'): -1}
quadratic = {('x0', 'x1'): 2, ('x0', 'x2'): 2, ('x1', 'x2'): 2}
 
Q = dict(linear)
Q.update(quadratic)
 
response = EmbeddingComposite(DWaveSampler(token=setting.tokencode)).sample_qubo(Q, num_reads=1000)
 
for sample, energy, num_occurrences, chain_break_fraction in list(response.data()):
    print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
