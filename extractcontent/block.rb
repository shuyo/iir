#!/usr/bin/ruby -Ku

require 'rubygems'
require 'hpricot'

TAGLIST = ['a','body','br','font','form','h','iframe','img','input','li','noscript','object','option','p','scr','script','select','span','table','th','tr','ul','abbr','address','area','b','blink','caption','center','dd','del','div','dl','dt','em','embed','file','head','hr','i','ifr','kbd','keeper','label','left','map','marquee','meta','ol','pairlist','param','q','rdf','small','smoothie','strike','strong','style','sup','tbody','td','textarea','u','acronym','bd','big','blockquote','bm','bodylang','button','cite','col','d','fieldset','frame','frameset','if','ld','legend','link','linklist','nobr','noframes','o','objlists','parms','rb','rp','rt','ruby','s','sc','thead','tt','var','w','wbr','zu','aa','abstract','acookie','actarray','action','ad','ads','adv','advanced','advancedselector','agent','ahref','alblen','align','alist','apan','applet','archive','arimgname','arr','array','arraycnt','articleinf','ary','asx','atr','author','autoload','azebu','background','bannersizes','base','basefont','bdo','bl','bodybackground','bold','border','bu','c','ca','cake','canvas','category','ccs','ceter','changeimages','charset','checkboxlist','checkimage','child','childelementslength','class','cnt','code','colgroup','color','com','comment','commercial','config','cont','content','control','controlcoverage','controls','cookie','cookielist','cookiequeries','cookies','copyright','corp','cript','csaction','csactiondict','csactionitem','csactions','csinit','csobj','csscriptdict','cssdropdownroot','darea','data','date','dc','defaultbuttons','delicious','dfn','dh','di','difi','display','distance','doccontent','document','domainname','domestic','dongura','dp','dropmenuobj','dtr','e','el','elements','els','emailriddlerarray','entry','esi','exkw','f','fb','feed','ffelements','fid','fileno','fjtignoreurl','fm','fn','foaf','fontclass','fontcolor','fontsize','founder','frm','fvn','g','global','google','hash','hdimg','hh','hspace','html','http','iargs','id','idarr','ids','ilayer','im','image','images','imgarray','imgsrc','in','ing','inittext','inputtype','ins','interviewdate','introtitle','invalidation','iskeyword','item','items','j','job','js','jsp','k','kbmj','keyword','kizasi','komatsu','krtgs','kwx','l','lastmenu','layer','leftedge','len','license','limit','lines','linkmore','links','linksno','lis','list','literal','ln','lst','lu','m','magazine','mail','mask','math','maxlen','messages','monte','mototext','msearch','msg','msgarr','mtifcommentsactive','mychar','mymonthtbl','mypageitem','mytblline','n','name','names','navroot','nazo','nbsp','ncnt','nickname','nodeid','nodename','nodes','noembed','noindex','nolayer','normal','notices','notrim','nowevent','nu','num','numsht','obentries','obj','objadd','objdiv','objentries','objexpert','objfocus','objimages','objmarks','objmore','optgroup','optionarray','optioncount','opts','pageend','pagelinkslen','pairs','par','path','pdcform','permits','pg','photocount','place','pre','price','processlibrary','profile','proflan','ptn','published','pwd','qs','question','r','radiolength','ratesrows','recobj','ref','rep','requires','results','right','rnd','roottag','samp','sav','scri','scrip','scripts','selarr','selects','sex','sfels','shm','side','siteid','siteurls','size','slstknd','smd','smr','snbuttons','snum','spa','spacer','spna','sptag','st','stkstknd','stock','str','string','strnumber','sub','subtitle','surfnavctx','t','tab','tablewidth','tboday','tc','temp','test','textflow','textformat','texts','tf','tfoot','tg','tgnms','tgsearch','this','ths','thumimgs','tipobj','title','tktgs','tli','tmp','tmpl','tn','tokutei','tool','total','txf','txp','txt','ullist','updated','url','users','v','valarr','valign','valuecommerce','values','vars','venter','view','vint','web','webparts','wgaftercontentandjs','which','word','work','x','xa','xml','xmp','xp','y','yabb','z','zaiko','zsr'
]

SEPARATORS = {
  /<!--\s*google_ad_section_start/ => :body,
  /<!--\s*google_ad_section_end/ => :unclassified,
  /<!--\s*extractcontent_body/ => :body,
  /<!--\s*extractcontent_abstract/ => :abstract,
  /<!--\s*extractcontent_header/ => :header,
  /<!--\s*extractcontent_footer/ => :footer,
  /<!--\s*extractcontent_comment/ => :comment,
  /<!--\s*extractcontent_menu/ => :menu,
  /<!--\s*extractcontent_linklist/ => :linklist,
  /<!--\s*extractcontent_end/ => :unclassified,
}

class ExtractFeature
  def initialize
    @all_tags = Hash.new(0)
  end
  attr_reader :all_tags

  def analyze(filename)
    html = open(filename){|f| f.read}

    if html =~ /<\/head\s*>/im
      html = $' #'
    end

    list = html.split(/<\/?(?:div|center|td)[^>]*>|<p\s*[^>]*class\s*=\s*["']?(?:posted|plugin-\w+)['"]?[^>]*>/)
    #puts list.join("\n====================================================================\n")

    block_tags = Hash.new(0)
    current_class = :undefined
    result = []
    list.each do |block|

      # tag count
      block.scan(/<([A-Za-z]+)/) do |tag|
        tag = tag[0].downcase
        block_tags[tag] += 1
        @all_tags[tag] += 1
      end
      features = []
      TAGLIST.each_with_index{|tag, i| features << "#{i}:#{block_tags[tag]}" if block_tags.key?(tag) }

      # detect class
      post_class = now_class = nil
      SEPARATORS.each do |regsep, clz|
        if regsep =~block
          if $`.length > $'.length
            $stderr.puts "specified already post_class at #{filename}" if post_class
            post_class = clz
          else
            $stderr.puts "specified already now_class at #{filename}" if now_class
            now_class = clz
          end
        end
      end

      current_class = now_class if now_class
      result << {:class=>current_class, :features=>features}
      current_class = post_class if post_class
    end
    result
  end
end

extractor = ExtractFeature.new
ARGV.each do |dir|
  Dir.foreach(dir) do |filename|
    next unless filename =~ /http%3A%2F%2F/
    data = extractor.analyze dir + '/' + filename
    data.each do |item|
      puts "#{item[:class]} #{item[:features].join(' ')}"
    end
  end
end

puts "all tags = \n#{extractor.all_tags.to_a.sort{|a,b|b[1]<=>a[1]}.map{|a|"#{a[0]},#{a[1]}"}.join("\n")}"


=begin
### TODO
- external link
- nofollow link
- using div id
- img alt
=end
