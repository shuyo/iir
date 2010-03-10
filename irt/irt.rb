#!/usr/bin/ruby
require 'zlib'

THRESHOLD = 1.2

learnfile = ARGV[0] || 'irt.data'
users = Hash.new
words = Hash.new
begin
  open(learnfile) do |f|
    users, words = Marshal.load(f)
  end
rescue
  puts "create new learning data randomizely."
end

data = []
Zlib::GzipReader.open('word_scores.txt.gz') do |f|
  while line = f.gets
    if line =~ /^([0-9]+)\s+([0-9]+)\s*([0-9\.]+)$/
      user_id = $1.to_i
      word_id = $2.to_i
      point = $3.to_f
      #next if point > 10000

      t = if point < THRESHOLD then 1 else 0 end
      data << [user_id, word_id, t]

      users[user_id] = rand unless users.key?(user_id)
      words[word_id] = rand unless words.key?(word_id)
    end
  end
end

10000.times do |k|
  eta = 0.01 #1.0/(k+10)
  e = 0
  error = 0
  data.sort_by{rand}.each do |user_id, word_id, t|
    z = users[user_id] - words[word_id]
    y = 1.0/(1.0+Math.exp(-z))
    e -= if t==1 then Math.log(y) else Math.log(1-y) end
    error += 1 if (t==1 && y<0.5) || (t==0 && y>0.5)

    grad_e_eta = eta*(y - t)
    users[user_id] -= grad_e_eta
    words[word_id] += grad_e_eta
  end
  puts "#{k}: #{error}, #{e}"
  open(learnfile+".1", 'w'){|f| Marshal.dump([users,words], f) } if (k % 50) == 0
end

open(learnfile, 'w'){|f| Marshal.dump([users,words], f) }

