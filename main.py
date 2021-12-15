import numpy as np

def sym_mult(a1, a2):
    if a1 == 'I2':
        return a2
    if a2 == 'I2':
        return a1
    if a1 == a2:
        return 'I2'

    # with X
    if a1 == 'X' and a2 == 'Y':
        return 'Z'
    if a1 == 'Y' and a2 == 'X':
        return 'Z'
    if a1 == 'X' and a2 == 'Z':
        return 'Y'
    if a1 == 'Z' and a2 == 'X':
        return 'Y'

    # with Y
    if a1 == 'Y' and a2 == 'Z':
        return 'X'
    if a1 == 'Z' and a2 == 'Y':
        return 'X'

def format_triple_tensor(m):
    return '{0}⊗{1}⊗{2}'.format(m[0], m[1], m[2])

def triple_tensor_dot(m):
    if type(m[0]) == str:
        return np.kron(np.kron(globals()[m[0]], globals()[m[1]]), globals()[m[2]])
    else:
        return np.kron(np.kron(m[0], m[1]), m[2])

def print_tensor_triplets(ms):
    ms = [format_triple_tensor(x) for x in ms]
    print(*ms, sep=', ')


# problem 1 ############################################################################################################
I2 = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
a = [1, -1, 1j, -1j]
symbolic_dict = {I2.tobytes(): 'I2', X.tobytes(): 'X', Y.tobytes(): 'Y', Z.tobytes(): 'Z'}
candidates = symbolic_dict.values()

S = [np.kron(np.kron(I2, I2), I2),
     np.kron(np.kron(X, X), X),
     np.kron(np.kron(X, Z), Y),
     np.kron(np.kron(I2, Y), Z)]
S_sym = [('I2', 'I2', 'I2'), ('X', 'X', 'X'), ('X', 'Z', 'Y'), ('I2', 'Y', 'Z')]  # symbolic representation of S
S_perp = set()

for e1 in candidates:
    for e2 in candidates:
        for e3 in candidates:
            for factor in a:
                A = factor * triple_tensor_dot((e1, e2, e3))
                if np.all([np.all(A @ A_prime == A_prime @ A) for A_prime in S]):
                    cs = [e1, e2, e3]
                    S_perp.add(tuple(cs))
S_perp = list(S_perp)
S_perp.sort()
print('S⊥ is given as: ')
print_tensor_triplets(S_perp)

# problem 2 ############################################################################################################

A_t = ('I2', 'X', 'I')
S_A_t = [[sym_mult(a1, ss) for a1, ss in zip(A_t, s) ]for s in S_sym]
S_A_t.sort()
print('With A1 {0}, we have S\'s coset with A1:'.format(format_triple_tensor(A_t)), end='')
print_tensor_triplets(S_A_t)


A_t = ('I2', 'Z', 'I2')
S_A_t = [[sym_mult(a1, ss) for a1, ss in zip(A_t, s) ]for s in S_sym]
S_A_t.sort()
print('With A1 {0}, we have S\'s coset with A1:'.format(format_triple_tensor(A_t)), end='')
print_tensor_triplets(S_A_t)


A_t = ('Z', 'I2', 'I2')
S_A_t = [[sym_mult(a1, ss) for a1, ss in zip(A_t, s) ]for s in S_sym]
S_A_t.sort()
print('With A1 {0}, we have S\'s coset with A1:'.format(format_triple_tensor(A_t)), end='')
print_tensor_triplets(S_A_t)


print('Together with S, given by ', end='')
print_tensor_triplets(S_sym)
print('It makes the complete S⊥')
print_tensor_triplets(S_perp)

# problem 3 ############################################################################################################

for i, A_n in enumerate(S_sym):
    coset = []
    for A_M in S_perp:
        cs = [sym_mult(a, am) for a, am in zip(A_n, A_M)]
        coset.append(format_triple_tensor(cs))
    print('{0}th coset of S⊥ with A^{0} {1} from S: '.format(i+1, format_triple_tensor(A_n)), end='')
    print(*coset, sep=', ')


# Problem 4
# find the S cosets
Ps = []  # store all P's
Ps_bytes = set()
temp = np.zeros((8, 8))
for A in S:
    temp = np.add(temp, A)
P1 = Pn = temp / len(S)
Ps.append(P1)

S_cs = []
for e1 in candidates:
    for e2 in candidates:
        for e3 in candidates:
            S_cs.append((e1, e2, e3))
S_cs = [triple_tensor_dot(x) for x in S_cs if x not in S_sym and x not in S_perp][:3]
for i, A_x in enumerate(S_cs):  # A_ is the current leader of the coset
    Pn = A_x  @ P1 @ A_x
    Ps.append(Pn)

# check orthogonality
for P_i in Ps:
    for P_j in Ps:
        if not np.all(P_i == P_j):
            assert not np.any(np.matmul(P_i, P_j))
temp = np.zeros((8, 8))
for x in Ps:
    temp = np.add(temp, x)
print('Summation of all P is: ' + str(temp))

with np.printoptions(precision=2, suppress=True):
    for i, P in enumerate(Ps):
        print('Projector {0}: '.format(i), end='\n')
        print(str(P).replace('0.  +0.j', '0.'))


# problem 6 ############################################################################################################
v1 = np.zeros((8, 1))
v1[0, 0] = 1
v2 = np.zeros((8, 1))
v2[2, 0] = 1
v3 = np.zeros((8, 1))
v3[5, 0] = 1
v4 = np.zeros((8, 1))
v4[7, 0] = 1

v = (1/2) * (v1 + 1j * v2 + 1j * v3 + v4)

assert np.all(Ps[0] @ v == v)

# problem 7 ############################################################################################################
for i, p in enumerate(Ps):
    print('Measureing with P_{0}, gives probability {1}'.format(i, v.transpose().conjugate() @ p @ v))

triple_tensor_dot(('I2', 'X', 'I2')) @ v1
triple_tensor_dot(('I2', 'X', 'I2')) @ v2
triple_tensor_dot(('I2', 'X', 'I2')) @ v3
triple_tensor_dot(('I2', 'X', 'I2')) @ v4
