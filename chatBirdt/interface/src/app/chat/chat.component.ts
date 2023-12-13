import { Component, OnInit } from '@angular/core';

// { Hero } from '../hero';
import { LlamaService } from '../llama.service';

@Component({
  selector: 'app-heroes',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})


export class ChatComponent implements OnInit {

  chat = [
    {type: "response", text:'Hallo, can I help You?', hasSpace:'hasNoSpace'},
  ];

  loading = false;

  constructor(private llamaService: LlamaService)
    {
      console.log('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

      this.llamaService.test()
        .subscribe(
          (response) => {
            console.log(response.data)

          })
    }

  ngOnInit(): void {}

  onClickMe() {
    let text = (<HTMLInputElement>document.getElementById('interaction')).value

    if(text !== '' && text.replace(/\s/g, '').length) {

      let hasSpace_ = 'hasNoSpace'
      if (text.indexOf('\n') >= 0) hasSpace_ = 'hasSpace';
      this.chat = this.chat.concat([{type: "input", text: text, hasSpace:hasSpace_}]);
      (<HTMLInputElement>document.getElementById('interaction')).value = '';
      let element = (<HTMLInputElement>document.getElementById("item1"));
      element.scrollTop = element.scrollHeight;

      let texts :string[] = []
      let fromAI :string[] = []
      this.chat.forEach(element => {
        texts.push(element.text);
        fromAI.push(element.type);
        //console.log(element.type)
      })

      this.loading = true;

      this.llamaService.chatDocument(text, texts, fromAI)
        .subscribe(
          (response) => {
                  console.log(response.data)

                  let hasSpace = 'hasNoSpace'
                  if (response.data.indexOf('\n') >= 0) hasSpace = 'hasSpace';
                  this.chat = this.chat.concat([{type: "response", text: response.data, hasSpace:hasSpace}]);
                }, (error) => {
                  console.log(error)
                  this.chat = this.chat.concat([{type: "response", text: error, hasSpace:'hasNoSpace'}]);
                },
                 () => {
                   this.loading = false;
          })
    }


  }


}
