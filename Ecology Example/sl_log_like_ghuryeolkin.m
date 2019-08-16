function f = sl_log_like_ghuryeolkin(s,muhat,sighat,m)
    p = length(s);
    temp = (m-1)*sighat- (s-muhat)'*(s-muhat)/(1-1/m);
    [~,a] = chol(temp);
    
    if (a == 0) % then positive definite
        result = -p/2*log(2*pi)+wcon(p,m-2)-wcon(p,m-1)-p/2*log(1-1/m);
        result = result-(m-p-2)/2*(log(m-1)+logdet(sighat));
        f = result + (m-p-3)/2*logdet(temp);
    else
        f = -Inf;
    end
    

function f = wcon(k,nu)
    f = -k*nu/2*log(2)-k*(k-1)/4*log(pi)-sum(gammaln(0.5*(nu-(1:k)+1)));

    
function y = logdet(A)

U = chol(A);
y = 2*sum(log(diag(U)));


