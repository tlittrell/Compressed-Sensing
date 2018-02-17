using JuMP,Gurobi

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

function compressedSensing(A,b,g)

    m = size(A,1)
    n = size(A,2)

    s0 = ones(n)
    c0 = fOfS(A,s0,b,g)
    dc0 = fgradient(A,s0,b,g)

    model = Model(solver = GurobiSolver(TimeLimit = 120, LazyConstraints = 1))

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


function LassoCompressedSensing(A,b)

    m = size(A,1)
    n = size(A,2)

    model = Model(solver = GurobiSolver(TimeLimit = 120))

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


m = 10
n = 20
k = 10
A = randn(m,n)
x0 = vcat(randn(k),zeros((n-k)))
b = A*x0

result = compressedSensing(A,b,0.01)
norm(result,0)
find(result)

resultLasso = LassoCompressedSensing(A,b)
norm(resultLasso,0)
find(resultLasso)