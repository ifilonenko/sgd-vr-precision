"""

This struct is responsible for leveraging the SVRG algorithm. The input for
the algorithm are the following:
Inputs:
   a :: Real, the alpha value which correlates to step size
   T :: Integer, number of iterations of applying random sampling of gradient
   K :: Integer, number of epochs of applying a full gradient

   w0 :: Array{Float64,1}, the initial randomized iterate
   wopt :: Array{Float64,1}, optimal w value take for distance purposes
   X :: Array{Float64,2}, training X values (input features)
   Y :: Array{Float64,1}, training Y values (labels)
   g_l :: (generic function) :: gradient loss function you wish to minimize
      This function must take in (wtilde,i)
         - wtilde, new w value per epoch
         - i,index

# Examples
```julia-repl
julia> SVRG{alpha,niters,nepochs}(w0,wopt,X,Y,g_l)
...
```
"""
struct SVRG{a,T,K} <: NonQuantized{a,T,K}
   SVRG{a,T,K}(w0::A,wopt::A,X::AT,Y::A,g_l) where
      {A<:Array{Float64,1}, AT<:Array{Float64,2},a,T,K} =
      run_algo(SVRG{a,T,K},w0,wopt,X,Y,g_l)
end
function run_algo(::Type{SVRG{a,T,K}},w0,wopt,X,Y,g_l) where {a,T,K}
      w = w0
      (N, d) = size(X)
      dist_to_optimum = zeros(T*K)
      for k = 1:K
         wtilde = w;
         gtilde = mapreduce(i->g_l(wilde,i), +, 1:N)
         w = wtilde;
         for t = 1:T
            i = rand(1:N)
            w = w - a*(g_l(w,i) - g_l(wtilde,i) + gtilde)
            dist_to_optimum[k + (epi-1)*niters] = norm(w - wopt);
         end
      end
      return (w, dist_to_optimum)
end
