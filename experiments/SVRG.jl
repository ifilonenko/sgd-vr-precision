"""

This struct is responsible for leveraging the SVRG algorithm. The input for
the algorithm are the following:
Inputs:
   a :: Real, the alpha value which correlates to step size
   T :: Integer, number of iterations of applying random sampling of gradient
   K :: Integer, number of epochs of applying a full gradient

   w0 :: Array{Float64}, the initial randomized iterate
   wopt :: Array{Float64}, optimal w value take for distance purposes
   X :: Array{Float64}, training X values (input features)
   Y :: Array{Float64}, training Y values (labels)
   g_l :: (generic function) :: gradient loss function you wish to minimize
      This function must take in (phi,i,X,Y)
         - phi,the full gradient
         - i,index
         - X,training features
         - Y,training labels

# Examples
```julia-repl
julia> SVRG{alpha,niters,nepochs}(w0,wopt,X,Y,g_l)
...
```
"""
struct SVRG{a,T,K} <: NonQuantized{a,T,K}
   SVRG{a,T,K}(w0::A,wopt::A,X::A,Y::A,g_l) where {A <:Array,a,T,K} =
      run_algo(SVRG{a,T,K},w0,wopt,X,Y,g_l)
end
function run_algo(::Type{SVRG{a,T,K}},w0,wopt,X,Y,g_l) where {a,T,K}
      w = w0
      (N, d) = size(X)
      dist_to_optimum = zeros(T*K)
      for k = 1:K
         wtidle = w;
         phi = map(i -> X[i,:]'*wtidle, 1:N)
         gtilde = mapreduce(i->g_l(phi,i,X,Y)*X[i,:], +, 1:N)
         for t = 1:T
            i = rand(1:N)
            w = w - a*(g_l(w,i,X,Y) - g_l(wtilde,i,X,Y) + gtilde)
            dist_to_optimum[k + (epi-1)*niters] = norm(w - wopt);
         end
      end
      return (w, dist_to_optimum)
end
