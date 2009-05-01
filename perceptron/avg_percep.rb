#!/usr/bin/ruby

traindata = [  # AND
  [[0, 0], -1],
  [[1, 0], -1],
  [[0, 1], -1],
  [[1, 1], +1],
]

degree = traindata[0][0].size + 1
w = Array.new(degree, 0)

20.times do |c|
  # shuffle
  traindata = traindata.sort_by{rand}

  # training
  n_errors = 0
  w_a = Array.new(degree, 0)
  n = 0
  traindata.each do |x, t|
    px = [1] + x # phai(x)
    s = 0        # sigma w^T phai(x_n)
    px.each_with_index do |x_i, i|
      s += w[i] * x_i
    end
    if s * t <= 0  # error
    #if (t>0)?(s<0):(s>=0) # 0 is also positive.
      puts [c+1, w, px, s, t].inspect
      n_errors += 1
      px.each_with_index do |x_i, i|
        w[i] += t * x_i
        w_a[i] += t * x_i * n
      end
    end
    n += 1
  end
  w_a.each_with_index do |w_i, i|
    w[i] -= w_i.to_f / n
  end

  if n_errors == 0
    puts "convergence: #{c}"
    break
  end
end

puts "w= #{w.inspect}"

