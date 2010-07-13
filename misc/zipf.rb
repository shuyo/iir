#!/usr/bin/ruby

list = " etaonisrhdlucmfwgypbvkxjqz".split(//)
prob = if ARGV[0] == "unif"
  [0.2] + Array.new(26){ 0.8 / 26.0 }
elsif ARGV[0] == "linear"
  (1..27).map{|i| (28.0 - i) / (14 * 27) }
elsif ARGV[0] == "zipf"
  (1..27).map{|i| 0.256973175704523 / i }
else
  [0.1918182,0.1041442,0.0729357,0.0651738,0.0596302,0.0564513,0.0558094,0.0515760,0.0497563,
   0.0492888,0.0349835,0.0331490,0.0225134,0.0217339,0.0202124,0.0197881,0.0171272,0.0158610,
   0.0145984,0.0137645,0.0124248,0.0082903,0.0050529,0.0013692,0.0009033,0.0008606,0.0007836]
end
name = ARGV[0] || "orig"
size = (ARGV[1] || 5000000).to_i

prob_sum = 0
cum_prob = []
prob.each do |x|
  cum_prob << prob_sum
  prob_sum += x
end
cum_prob << 1.0

module R;def self.rand;Kernel::rand;end;end
random = Random.new rescue R

map = Hash.new(0)
word = ""
while true
  r = random.rand
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
      break if map.size == size
    end
    word = ""
  else
    word += x
  end
end

open("#{name}#{map.size/1000}k.txt", "w") do |f|
  f.puts "rank,word,freq,rank*freq,freq_freq"
  freq = rank = 1
  map.to_a.sort_by{|x| -x[1]}.each_with_index do |x, r|
    if freq != x[1]
      f.puts "#{rank},#{x[0]},#{x[1]},#{rank*x[1]},#{r-rank+2}" if rank>0
      freq = x[1]
      rank = r + 2
    end
  end
end

