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
      This function must take in (wtilde,X,Y)
         - wtilde, new w value per epoch
         - X, training data
         - Y, training labels

# Examples
```julia-repl
julia> SVRG{alpha,niters,nepochs}(w0,wopt,X,Y,g_l)
...
```
"""
struct SVRG{a,T,K} <: NonQuantized{a,T,K}
   SVRG{a,T,K}(w0::AbstractArray{F,N},wopt::AbstractArray{F,N},
      X::AbstractArray{F,N2},Y::AbstractArray{F2,N},g_l) where
      {F<:Number,F2<:Number,a,T,K,N,N2} =
      run_algo(SVRG{a,T,K},w0,wopt,X,Y,g_l)
end
function run_algo(::Type{SVRG{a,T,K}},w0,wopt,X,Y,g_l) where {a,T,K}
      w = w0
      (N, d) = size(X)
      dist_to_optimum = zeros(T*K)
      for k = 1:K
         wtilde = w
         gtilde = (mapreduce(i->g_l(wtilde,X[i,:]',Y[i]), +, 1:N) / N)[1,:]
         w = wtilde
         for t = 1:T
            i = rand(1:N)
            xi = X[i,:]';
            yi = Y[i];
            w = w - a*(g_l(w,xi,yi) - g_l(wtilde,xi,yi) + gtilde)
            dist_to_optimum[t+((k-1)*T)] = norm(w - wopt)
         end
      end
      return (w, dist_to_optimum)
end
