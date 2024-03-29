using JuMP,	Gurobi, DataFrames, Gadfly, DataFramesMeta, Plots

function fgradient(A,s,b,g)
    S = Diagonal(s)
    n = size(S,1)
    Kinv = pinv(A*S*A')
    alpha = (-1/(g)) * Kinv * b
    #return [(1 + b' * (1/(2*g)) * -Kinv * A[:,i] * transpose(A[:,i]) * Kinv * b) for i=1:n]
    return [(1 - g/2*dot(A[:,j],alpha)^2) for j=1:n]
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

    @assert n > m

    s0 = ones(n)
    c0 = fOfS(A,s0,b,g)
    dc0 = fgradient(A,s0,b,g)

    model = Model(solver = GurobiSolver(TimeLimit = 120, LazyConstraints = 1,
					OutputFlag = 1*verbose))

    @variable(model, s[1:n],Bin)
    @variable(model, t>=0)

    @objective(model, Min, t)

    cutCount = 1
    #bestObj = c0
    #bestSolution = s0[:]
    @constraint(model, t >= c0 + dot(dc0,s-s0))

    function outer_approximation(cb)
        cutCount += 1
        c = fOfS(A,getvalue(s),b,g)
        dc = fgradient(A,getvalue(s),b,g)
        #if c < bestObj
        #    bestObj = c
        #    bestSolution = getvalue(s)[:]
        #end
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

    @assert n > m

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
	return @linq DataFrame(Algo = ["Dual", "Lasso"], M = [m, m], N = [n,n], K = [k,k],
					 Gamma = [g, NaN], Sims = [numSims, numSims],
					 Accuracy = acc/numSims, FalsePositive = fp/numSims,
					 ZeroNorm = zeroNorm/numSims) |>
                 transform(KdivM = :K./:M, MdivN = :M./:N)
end

function simulate_multiple(mlist, nlist, klist, glist, numSims)
	result = DataFrame()
	for m in mlist
		for n in nlist
			for k in [x for x in klist if x <= m]
				for g in glist
                    try
                        @show [m,n,k,g]
					    result = vcat(result, simulate(m,n,k,g,numSims))
                    catch
                        println()
                        println("The following simulation failed")
                        @show [m,n,k,g]
                        println()
                    end
				end
			end
		end
	end
	return result
end

function draw_pictures(;mlist=[0], nlist=[0], klist=[0], glist=[0], numSims=0)
    data = simulate_multiple(mlist, nlist, klist, glist, numSims)

    plt1 = plot(data, x = :KdivM, y = :Accuracy, Geom.line, color = :Algo);
    plt2 = plot(data, x = :KdivM, y = :FalsePositive, Geom.line, color = :Algo);

    plt3 = plot(data, x = :MdivN, y = :Accuracy, Geom.line, color = :Algo);
    plt4 = plot(data, x = :MdivN, y = :FalsePositive, Geom.line, color = :Algo);

    if size(klist,1) > 1
        draw(PNG("Accuracy KdivM (m=$mlist n=$nlist g=$glist k=$klist).png", 3inch, 3inch), plt1)
        draw(PNG("False Positive KdivM (m=$mlist n=$nlist g=$glist k=$klist).png", 3inch, 3inch), plt2)
    elseif size(nlist,1) > 1
        draw(PNG("Accuracy MdivN (m=$mlist n=$nlist g=$glist k=$klist).png", 3inch, 3inch), plt3)
        draw(PNG("False Positive MdivN (m=$mlist n=$nlist g=$glist k=$klist).png", 3inch, 3inch), plt4)
    end
end


numSims = 40

###################################

klist = [2,4,6,8,10]
mlist = [10]
nlist = [15]
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.001],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.01],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.1],numSims=numSims)


klist = [5]
mlist = [10]
nlist = [10,15,25,50,75,100]
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.001],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.01],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.1],numSims=numSims)


###################################
scale = 5

klist = [2,4,6,8,10]*scale
mlist = [10]*scale
nlist = [15]*scale
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.001],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.01],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.1],numSims=numSims)


klist = [5]*scale
mlist = [10]*scale
nlist = [10,15,25,50,75,100]*scale
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.001],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.01],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.1],numSims=numSims)


###################################
scale = 10

klist = [2,4,6,8,10]*scale
mlist = [10]*scale
nlist = [15]*scale
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.001],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.01],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.1],numSims=numSims)


klist = [5]*scale
mlist = [10]*scale
nlist = [10,15,25,50,75,100]*scale
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.001],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.01],numSims=numSims)
draw_pictures(mlist=mlist,nlist=nlist,klist=klist,glist=[0.1],numSims=numSims)
