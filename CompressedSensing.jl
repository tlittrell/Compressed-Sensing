using JuMP,	Gurobi, DataFrames

function fgradient(A,s,b,g)
    S = Diagonal(s)
    Kinv = pinv(A*S*A')
    return [(1 + b' * (1/(2*g)) * -Kinv * A[:,i] * transpose(A[:,i]) * Kinv * b) for i=1:n]
end

function fOfS(A,s,b,g)
    S = Diagonal(s)
    n = size(S,1)
    e = ones(n)
    Kinv = pinv(A*S*A')
    return e'*S*e + (1/(2*g)) * b'*Kinv*b
end

function compressedSensing(A,b,g; verbose = 0)

    m = size(A,1)
    n = size(A,2)

    s0 = ones(n)
    c0 = fOfS(A,s0,b,g)
    dc0 = fgradient(A,s0,b,g)

    model = Model(solver = GurobiSolver(TimeLimit = 120, LazyConstraints = 1,
					OutputFlag = 1*verbose))

    @variable(model, s[1:n],Bin)
    @variable(model, t>=0)

    @objective(model, Min, t)

    # ensure that our matrix is invertible
    #@constraint(model, sum(s) >= m)

    cutCount = 1
    bestObj = c0
    bestSolution = s0[:]
    @constraint(model, t >= c0 + dot(dc0,s-s0))

    function outer_approximation(cb)
        cutCount += 1
        c = fOfS(A,getvalue(s),b,g)
        dc = fgradient(A,getvalue(s),b,g)
        if c < bestObj
            bestObj = c
            bestSolution = getvalue(s)[:]
        end
        @lazyconstraint(cb, t >= c + dot(dc, s-getvalue(s)))
    end
    addlazycallback(model, outer_approximation)

    status = solve(model)
    deltaT = getsolvetime(model)

    if status != :Optimal
        Gap = 1-JuMP.getobjbound(model) / getobjectivevalue(model)
    end

    if status == :Optimal
        bestSolution = getvalue(s)[:]
    end
    return getvalue(s)
end


function LassoCompressedSensing(A,b; verbose = 0)

    m = size(A,1)
    n = size(A,2)

    model = Model(solver = GurobiSolver(TimeLimit = 120, OutputFlag = 1*verbose))

    @variable(model, xPlus[1:n] >= 0)
    @variable(model, xMinus[1:n] >= 0)
    @variable(model, x[1:n])
    @variable(model, t>=0)

    @objective(model, Min, t)

    @constraint(model, A*(xPlus - xMinus) .== b)
    @constraint(model,t >= sum(xPlus) + sum(xMinus))
    @constraint(model, x .== xPlus - xMinus)

    solve(model)
    return getvalue(x)
end

function accuracy(result,k)
	return size([x for x in find(result) if x <= k],1)/k
end

function falsePositive(result, k)
	return size([x for x in find(result) if x > k],1) / size(find(result),1)
end

function simulate(m,n,k,g,numSims)
	acc = zeros(2)
	fp = zeros(2)
	zeroNorm = zeros(2)
	for i = 1:numSims
		A = randn(m,n)
		x0 = vcat(randn(k),zeros((n-k)))
		b = A*x0

		results = [compressedSensing(A,b,g), LassoCompressedSensing(A,b)]
		acc += map(x -> accuracy(x,k), results)
		fp += map(x -> falsePositive(x,k), results)
		zeroNorm += map(x -> norm(x,0), results)
	end
	return DataFrame(Algo = ["Dual", "Lasso"], M = [m, m], N = [n,n], K = [k,k],
					 Gamma = [g, NaN], Sims = [numSims, numSims],
					 Accuracy = acc/numSims, FalsePositive = fp/numSims,
					 ZeroNorm = zeroNorm/numSims)
end

function simulate_multiple(mlist, nlist, klist, glist, numSims)
	result = DataFrame()
	for m in mlist
		for n in nlist
			for k in [x for x in klist if x <= m]
				for g in glist
					result = vcat(result, simulate(m,n,k,g,numSims))
				end
			end
		end
	end
	return result
end

m = 10
n = 30
k = 4
g = 0.01
numSims = 10
simulate(m,n,k,g,numSims)

temp1 = DataFrame()
temp2 = DataFrame(A=[3,4], B=[3,4])
print(vcat(temp1,temp2))

mlist = [10]
nlist = [30]
klist = [4]
glist = [0.01]
numSims = 10
simulate_multiple(mlist, nlist, klist, glist, numSims)
