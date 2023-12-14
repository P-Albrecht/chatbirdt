import { Component } from '@angular/core';
import {ActivatedRoute, Params, Router} from "@angular/router";
import { HttpClient } from '@angular/common/http';

import { FlaskapiService } from './flaskapi.service';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ChatBirdt';
  text = '....'

  constructor(private router: Router, private activatedRoute: ActivatedRoute, private http: HttpClient, private flaskapiService: FlaskapiService) {
  }

  async ngOnInit() {

  }

  onClickMe() {
    let t = (<HTMLInputElement>document.getElementById('interaction')).value
    this.flaskapiService.blank(t)
      .subscribe(
        (response) => {
          console.log(response)
          this.text = response.data
        },
        (error) => {
          console.log('oops', error)
          this.text = 'ERROR'
        }
      );
  }
}

