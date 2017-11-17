"""

This struct is responsible for leveraging the LM-HALP algorithm. The input for
the algorithm are the following:
Inputs:
   a :: Real, the alpha value which correlates to step size
   T :: Integer, number of iterations of applying random sampling of gradient
   K :: Integer, number of epochs of applying a full gradient
   R :: Scaled, a FixedPointNumbers.Scaled type use to quantize (red box)

   w0 :: Array{Float64}, the initial randomized iterate
   wopt :: Array{Float64}, optimal w value take for distance purposes
   X :: Array{Float64}, training X values (input features)
   Y :: Array{Float64}, training Y values (labels)
   b :: Int, number of bits to quantize (blue box bits)
   p :: Int, number of bits to quantize (purple box bits)
   mu :: Float64, scale factor
   tol :: Float64, tolerance from [0,1.0]

   g_l :: (generic function) :: gradient loss function you wish to minimize
      This function must take in (phi,i,X,Y)
         - phi,the full gradient
         - i,index
         - X,training features
         - Y,training labels

# Examples
```julia-repl
julia> R = Scaled{Int8,7,1.0/128.0,Randomized};
julia> b = 7;
julia> p = 7;
julia> mu = 0.0001;
julia> tol = 0.0001;
julia> LMHALP{alpha,niters,nepochs,R}(w0,wopt,X,Y,b,p,mu,tol,g_l)
...
```
"""
struct LMHALP{a,T,K,R <: Scaled} <: Quantized{a,T,K,R}
   LMHALP{a,T,K,R}(w0::AbstractArray{F,N},wopt::AbstractArray{F,N},
   X::AbstractArray{F,N2},Y::AbstractArray{F2,N},b,p,mu,tol,g_l) where
        {F<:Number,F2<:Number,R <: Scaled,a,T,K,N,N2} =
      run_algo(LMHALP{a,T,K,R},w0,wopt,X,Y,b,p,mu,tol,g_l)
end
function rounder(b::Integer)
    if b-1 <= maxf(Int8)
        return Int8
    elseif b-1 <= maxf(Int16)
        return Int16
    elseif b-1 <= maxf(Int32)
        return Int32
    else
        return Int64
    end
end
function run_algo(::Type{LMHALP{a,T,K,R}},w0,wopt,X,Y,b,p,mu,tol,g_l) where
   {R <: Scaled,a,T,K}
      w = w0
      (N, d) = size(X)
      z = zeros(d)
      dist_to_optimum = zeros(T*K)
      s = 1.0
      X = R(X)
      for k = 1:K
          w = w + z
          phi = map(i -> X[i,:]'*w, 1:N)
          gtilde = mapreduce(i->g_l(phi,X[i,:]',Y[i])*X[i,:]', +, 1:N)/N
          s = norm(gtilde)/(mu*(2^(b-1)-1))
          blue_box = Scaled{rounder(b),b-1,s,Randomized}
          p_s = (s./get_s(R))*2.0^(b-get_f(R)+1-p)
          purple_box = Scaled{rounder(p),p-1,p_s,Randomized}
          green_box = R*purple_box
          assert(subdomain(blue_box,green_box,tol))
          htilde = green_box(a*gtilde)
          z = zeros(blue_box,d)
          for t = 1:T
              i = rand(1:N)
              xi = X[i,:]'
              yi = Y[i]
              inner_l = g_l(phi+xi*z,xi,yi)
              inner_purple = a.*(inner_l-g_l(phi,xi,yi))
              i_g = green_box(purple_box(inner_purple)*xi)
              z = blue_box(float(green_box(float(z)) - (i_g - htilde)))
              dist_to_optimum[t+((k-1)*T)] = norm(z + wtilde - wopt);
          end
      end
      return (w, dist_to_optimum);
  end
