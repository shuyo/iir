#!/usr/bin/ruby

# generate *.mat/*.clabel for CLUTO

input = ARGV[0] || 'corpus'
output = input.sub(/\.[^\.]+/, '')

data = open(input){|f| Marshal.load(f) }
docs = data[:docs]
terms = data[:terms]

total = 0
terms.each do |term, map|
  total += map.size
end
termlist = terms.keys

open(output+".mat", "w") do |f|
  f.puts "#{docs.size} #{terms.size} #{total}"
  (0..(docs.size-1)).each do |doc_id|
    row = []
    termlist.each_with_index do |term, term_id|
      v = terms[term][doc_id]
      row << term_id+1 << v if v && v>0
    end
    f.puts row.join(" ")
  end
end

open(output+".clabel", "w") do |f|
  termlist.each do |term|
    f.puts term
  end
end

open(output+".rlabel", "w") do |f|
  docs.each do |doc|
    f.puts doc[:title]
  end
end




