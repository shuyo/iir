#!/usr/bin/ruby -Ku
$KCODE="u"

# Author:: Nakatani Shuyo
# Copyright:: (c)2007/2008 Cybozu Labs Inc. All rights reserved.
# License:: BSD

# = ExtractContent : Extract Content Module for html
# This module is to extract the text from web page ( html content ).
# Automatically extracts sub blocks of html which have much possibility that it is the text 
# ( except for menu, comments, navigations, affiliate links and so on ).

# == PROCESSES
# - separating blocks from html, calculating score of blocks and ignoring low score blocks.
# - for score calculation, using block arrangement, text length, whether has affiliate links or characteristic keywords
# - clustering continuous, high score blocks and comparing amang clusters
# - if including "Google AdSense Section Target", noticing it in particular

module ExtractContent
  # onvert from character entity references
  CHARREF = {
    '&nbsp;' => ' ',
    '&lt;'   => '<',
    '&gt;'   => '>',
    '&amp;'  => '&',
    '&laquo;'=> "\xc2\xab",
    '&raquo;'=> "\xc2\xbb",
  }

  # Default option parameters.
  DEFAULT = {
		:threshold => 100,																				# threhold for score of the text
		:min_length => 80,																				# minimum length of evaluated blocks
		:decay_factor => 0.73,																		# decay factor for block score
		:continuous_factor => 1.62, 															# continuous factor for block score ( the larger, the harder to continue )
		:punctuation_weight => 10,																# score weight for punctuations
		:punctuations => /(\343\200[\201\202]|\357\274[\201\214\216\237]|\.[^A-Za-z0-9]|,[^0-9]|!|\?)/,
																															# punctuation characters
		:waste_expressions => /Copyright|All Rights Reserved/i, 	# characteristic keywords including footer
		:debug => false,																					# if true, output block information to stdout
  }

  class Extractor < Module

    # Sets option parameters to default.
    # Parameter opt is given as Hash
    def initialize(opt=nil)
      @default = DEFAULT.clone
      @default.update(opt) if opt
    end

    # Analyses the given HTML text, extracts body and title.
    def analyse(html, opt=nil)
      # frameset or redirect
      return ["", extract_title(html)] if html =~ /<\/frameset>|<meta\s+http-equiv\s*=\s*["']?refresh['"]?[^>]*url/i

      # option parameters
      opt = if opt then @default.merge(opt) else @default end

      # header & title
      title = if html =~ /<\/head\s*>/im
        html = $' #'
        extract_title($`)
      else
        extract_title(html)
      end

      # Google AdSense Section Target
      html.gsub!(/<!--\s*google_ad_section_start\(weight=ignore\)\s*-->.*?<!--\s*google_ad_section_end.*?-->/m, '')
      if html =~ /<!--\s*google_ad_section_start[^>]*-->/n
        html = html.scan(/<!--\s*google_ad_section_start[^>]*-->.*?<!--\s*google_ad_section_end.*?-->/mn).join("\n")
      end

      # eliminate useless text
      html = eliminate_useless_tags(html)

      # heading tags including title
      html.gsub!(/(<h\d\s*>\s*(.*?)\s*<\/h\d\s*>)/is) do |m|
        if $2.length >= 3 && title.include?($2) then "<div>#{$2}</div>" else $1 end
      end

      # extract text blocks
      factor = continuous = 1.0
      body = ''
      score = 0
      bodylist = []
      list = html.split(/<\/?(?:div|center|td)[^>]*>|<p\s*[^>]*class\s*=\s*["']?(?:posted|plugin-\w+)['"]?[^>]*>/)
      list.each do |block|
        next unless block
        block.strip!
        next if has_only_tags(block)
        continuous /= opt[:continuous_factor] if body.length > 0

        # ignore link list block
        notlinked = eliminate_link(block)
        next if notlinked.length < opt[:min_length]

        # calculate score of block
        c = (notlinked.length + notlinked.scan(opt[:punctuations]).length * opt[:punctuation_weight]) * factor
        factor *= opt[:decay_factor]
        not_body_rate = block.scan(opt[:waste_expressions]).length + block.scan(/amazon[a-z0-9\.\/\-\?&]+-22/i).length / 2.0
        c *= (0.72 ** not_body_rate) if not_body_rate>0
        c1 = c * continuous
        puts "----- #{c}*#{continuous}=#{c1} #{notlinked.length} \n#{strip_tags(block)[0,100]}\n" if opt[:debug]

        # treat continuous blocks as cluster
        if c1 > opt[:threshold]
          body += block + "\n"
          score += c1
          continuous = opt[:continuous_factor]
        elsif c > opt[:threshold] # continuous block end
          bodylist << [body, score]
          body = block + "\n"
          score = c
          continuous = opt[:continuous_factor]
        end
      end
      bodylist << [body, score]
      body = bodylist.inject{|a,b| if a[1]>=b[1] then a else b end }
      [body[0], title]
    end

    # Extracts title.
    def extract_title(st)
      if st =~ /<title[^>]*>\s*(.*?)\s*<\/title\s*>/im
        strip_tags($1)
      else
        ""
      end
    end

    private

    # Eliminates useless tags
    def eliminate_useless_tags(html)
      # eliminate useless symbols
      html.gsub!(/\342(?:\200[\230-\235]|\206[\220-\223]|\226[\240-\275]|\227[\206-\257]|\230[\205\206])/,'')

      # eliminate useless html tags
      html.gsub!(/<(script|style|select|noscript)[^>]*>.*?<\/\1\s*>/imn, '')
      html.gsub!(/<!--.*?-->/m, '')
      html.gsub!(/<![A-Za-z].*?>/s, '')
      html.gsub!(/<div\s[^>]*class\s*=\s*['"]?alpslab-slide["']?[^>]*>.*?<\/div\s*>/m, '')
      html.gsub!(/<div\s[^>]*(id|class)\s*=\s*['"]?\S*more\S*["']?[^>]*>/is, '')

      html
    end

    # Checks if the given block has only tags without text.
    def has_only_tags(st)
      st.gsub(/<[^>]*>/imn, '').gsub("&nbsp;",'').strip.length == 0
    end

    # eliminates link tags
    def eliminate_link(html)
      count = 0
      notlinked = html.gsub(/<a\s[^>]*>.*?<\/a\s*>/imn){count+=1;''}.gsub(/<form\s[^>]*>.*?<\/form\s*>/imn, '')
      notlinked = strip_tags(notlinked)
      return "" if notlinked.length < 20 * count || islinklist(html)
      return notlinked
    end

    # determines whether a block is link list or not
    def islinklist(st)
      if st=~/<(?:ul|dl|ol)(.+?)<\/(?:ul|dl|ol)>/ism
        listpart = $1
        outside = st.gsub(/<(?:ul|dl)(.+?)<\/(?:ul|dl)>/ismn, '').gsub(/<.+?>/mn, '').gsub(/\s+/, ' ')
        list = listpart.split(/<li[^>]*>/)
        list.shift
        rate = evaluate_list(list)
        outside.length <= st.length / (45 / rate)
      end
    end

    # estimates how much degree of link list
    def evaluate_list(list)
      return 1 if list.length == 0
      hit = 0
      list.each do |line|
        hit +=1 if line =~ /<a\s+href=(['"]?)([^"'\s]+)\1/imn
      end
      return 9 * (1.0 * hit / list.length) ** 2 + 1
    end

    # Strips tags from html.
    def strip_tags(html)
      st = html.gsub(/<.+?>/m, '')
      # Convert from wide character to ascii
      st.gsub!(/\357\274([\201-\272])/){($1[0]-96).chr} # symbols, 0-9, A-Z
      st.gsub!(/\357\275([\201-\232])/){($1[0]-32).chr} # a-z
      st.gsub!(/\342[\224\225][\200-\277]/, '') # keisen
      st.gsub!(/\343\200\200/, ' ')
      CHARREF.each{|ref, c| st.gsub!(ref, c) }
      require 'cgi'
      st = CGI.unescapeHTML(st)
      st.gsub(/[ \t]+/, " ")
      st.gsub(/\n\s*/, "\n")
    end
  end

  include Extractor.new
end
