#!/usr/bin/ruby

list = "abcdefghijklmnopqrstuvwxyz ".split(//)
prob = [
 0.0651738,0.0124248,0.0217339,0.0349835,0.1041442,0.0197881,0.0158610,0.0492888,0.0558094,0.0009033,
 0.0050529,0.0331490,0.0202124,0.0564513,0.0596302,0.0137645,0.0008606,0.0497563,0.0515760,0.0729357,
 0.0225134,0.0082903,0.0171272,0.0013692,0.0145984,0.0007836,0.1918182
]

prob_sum = 0
cum_prob = [0]
prob.each do |x|
  prob_sum += x
  cum_prob << prob_sum
end

map = Hash.new(0)
word = ""
while true
  r = rand
  l = 0
  h = prob.size
  while h>l+1
    m = (h+l)/2
    if r < cum_prob[m]
      h = m
    else
      l = m
    end
  end
  x = list[l]

  if x == " "
    if word.length > 0
      map[word] += 1
      if map.size % 100000 == 0
        open("zipf#{map.size/1000}k.txt", "w") do |f|
          f.puts "rank,word,freq,rank*freq,freq_freq"
          freq = rank = 0
          map.to_a.sort_by{|x| -x[1]}.each_with_index do |x, r|
            if freq != x[1]
              f.puts "#{rank},#{x[0]},#{x[1]},#{rank*x[1]},#{r-rank+1}" if rank>0
              freq = x[1]
              rank = r + 1
            end
          end
        end
      end
    end
    word = ""
  else
    word += x
  end
end

